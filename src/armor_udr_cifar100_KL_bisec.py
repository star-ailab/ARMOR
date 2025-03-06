import argparse
import logging
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os

from resnet import *
from utils_udr import *
import scipy.optimize as opt

mu = torch.tensor(cifar100_mean).view(3, 1, 1).to(device)
std = torch.tensor(cifar100_std).view(3, 1, 1).to(device)


def normalize(X):
    return (X - mu)/std


upper_limit, lower_limit = 1,0


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        
        all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def attack_udr(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None, 
               lamda=None):
    max_loss = torch.zeros(y.shape[0]).to(device)
    max_delta = torch.zeros_like(X).to(device)
    for _ in range(restarts):
        delta = torch.zeros_like(X).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            
            tau = alpha
            loss = F.cross_entropy(output, y)
            loss_c = lamda / tau * 0.5*((delta-epsilon*torch.sign(delta))**2*(torch.abs(delta)>=epsilon)).sum()
            
            grad = torch.autograd.grad(loss, delta, retain_graph=True)
            grad_c = torch.autograd.grad(loss_c, delta)
            
            d = delta[index, :, :, :]
            g = grad[0][index, :, :, :]
            x = X[index, :, :, :]
            # Gradient ascent step (ref to Algorithm 1 - step 2bi in our paper)
            d = d + alpha * torch.sign(g) # equal x_adv = x_adv + alpha * torch.sign(g)
            
            g_c = grad_c[0][index, :, :, :]
            # Projection step (ref to Algorithm 1 - step 2bii in our paper)
            d = d - alpha * g_c
            
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            
        all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def mynorm(x, order): 
    x = torch.reshape(x, [x.shape[0], -1])
    b, d = x.shape 
    if order == 1: 
        return 1./d * torch.sum(torch.abs(x), dim=1) # [b,]
    elif order == 2: 
        return torch.sqrt(1./d * torch.sum(torch.square(x), dim=1)) # [b,]
    elif order == np.inf:
        return torch.max(torch.abs(x), dim=1)[0] # [b,]
    else: 
        raise ValueError

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ResNet')
    parser.add_argument('--l2', default=0, type=float)
    parser.add_argument('--l1', default=0, type=float)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-test', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr-schedule', default='piecewise', choices=['superconverge', 'piecewise', 'linear', 'piecewisesmoothed', 'piecewisezoom', 'onedrop', 'multipledecay', 'cosine', 'cyclic'])
    parser.add_argument('--lr-max', default=0.01, type=float)
    parser.add_argument('--lr-one-drop', default=0.01, type=float)
    parser.add_argument('--lr-drop-epoch', default=100, type=int)
    parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
    parser.add_argument('--epsilon', default=0.01, type=int)
    parser.add_argument('--attack-iters', default=10, type=int)
    parser.add_argument('--attack-iters-test', default=10, type=int)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--pgd-alpha', default=0.001, type=float)
    parser.add_argument('--fgsm-alpha', default=1.25, type=float)
    parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
    parser.add_argument('--fgsm-init', default='random', choices=['zero', 'random', 'previous'])
    parser.add_argument('--fname', default='cifar_model_udr', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--width-factor', default=10, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--cutout-len', type=int)
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup-alpha', type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--chkpt-iters', default=300, type=int)
    parser.add_argument('--awp-gamma', default=0.01, type=float)
    parser.add_argument('--awp-warmup', default=0, type=int)
    parser.add_argument('--lamda_init', default=1.0, type=float, help='initial value for lambda')
    parser.add_argument('--lamda_lr', default=2e-2, type=float, help='learning rate to update lambda')
    parser.add_argument('--lamda_period', default=10, type=int, help='period for updating lambda')
    parser.add_argument('--scale', default=2e-6, help='cost scale')
    
    ## armor params
    parser.add_argument('--eps', default=0.2, help='armor epsilon')
    parser.add_argument('--lambd', default=1.0, help='armor initial lambda')
    parser.add_argument('--t', default=0.75, help='natural vs. adv. loss (between 0 and 1)')
    parser.add_argument('--k_', default=0.12, help='lower-boud ARMOR loss via regularization')
    parser.add_argument('--verbose', action='store_true', default=False, help='print armor parameters after each minibatch')
    
    return parser.parse_args()

def main():
    args = get_args()

    model_name = 'ds_25=cifar100_model={}_method={}'.format(args.model, '_armor_udr_noawp_ours')

    if not os.path.exists(model_name):
        os.makedirs(model_name)

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler(os.path.join(model_name, 'eval.log' if args.eval else 'output.log')),
            logging.StreamHandler()
        ])

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    transforms = [Crop(32, 32), FlipLR()]
    if args.cutout:
        transforms.append(Cutout(args.cutout_len, args.cutout_len))
    if args.val:
        try:
            dataset = torch.load("cifar100_validation_split.pth")
        except:
            print("Couldn't find a dataset with a validation split, did you run "
                  "generate_validation.py?")
            return
        val_set = list(zip(transpose(dataset['val']['data']/255.), dataset['val']['labels']))
        val_batches = Batches(val_set, args.batch_size, shuffle=False, num_workers=2)
    else:
        dataset = cifar100(args.data_dir)
    train_set = list(zip(transpose(pad(dataset['train']['data'], 4)/255.),
        dataset['train']['labels']))
    train_set_x = Transform(train_set, transforms)
    train_batches = Batches(train_set_x, args.batch_size, shuffle=True, set_random_choices=True, num_workers=2)

    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    test_batches = Batches(test_set, args.batch_size_test, shuffle=False, num_workers=2)

    epsilon = args.epsilon
    pgd_alpha = args.pgd_alpha

    if args.model == 'ResNet':
        model = ResNet18(num_classes=100)
        proxy = ResNet18(num_classes=100)
    else:
        raise ValueError("Unknown model")

    model = nn.DataParallel(model).to(device)
    proxy = nn.DataParallel(proxy).to(device)
        
    if args.l2:
        decay, no_decay = [], []
        for name,param in model.named_parameters():
            if 'bn' not in name and 'bias' not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        params = [{'params':decay, 'weight_decay':args.l2},
                  {'params':no_decay, 'weight_decay': 0 }]
    else:
        params = model.parameters()

    ## armor
    def lambda_objective_derivative(epsilon,kappa,lambda_,loss_array,cost_array):
        max_loss=np.max((loss_array)/(lambda_+kappa)-cost_array)
        return epsilon+np.log(np.average(np.exp((loss_array)/(lambda_+kappa)-cost_array - max_loss))) + max_loss\
            -(lambda_+kappa)**(-1)*np.average(loss_array*np.exp((loss_array)/(lambda_+kappa)-cost_array-max_loss))/np.average(np.exp((loss_array)/(lambda_+kappa)-cost_array-max_loss))

    def optimize_lambda(epsilon,kappa,loss_array,cost_array,lambda_b=10,lambda_max=100):
        if lambda_objective_derivative(epsilon,kappa,0,loss_array,cost_array)>=0:
            return 0, True
        else:
            while lambda_objective_derivative(epsilon,kappa,lambda_b,loss_array,cost_array)<=0:
                lambda_b=2*lambda_b
                if lambda_b>=lambda_max:
                    return lambda_max, False
            lambda_star=opt.bisect(lambda z: lambda_objective_derivative(epsilon,kappa,z,loss_array,cost_array),0,lambda_b)
    
            return lambda_star, True

    class armor (nn.Module):
        def __init__(self):
            super(armor, self).__init__()
            self.lambda_ = args.lambd
        
        def forward(self, model, x_natural, x_adv, y, tau):
            delta = x_adv-x_natural
            losses = F.cross_entropy(model(normalize(x_adv)), y, reduction='none')
            costs = args.scale / tau * 0.5*((delta-epsilon*torch.sign(delta))**2*(torch.abs(delta)>=epsilon)).sum(dim=(1,2,3))
            self.lambda_, success = optimize_lambda(args.eps, args.k_, losses.detach().cpu().numpy(), costs.detach().cpu().numpy())
            maxLoss = (1.0/(self.lambda_+ args.k_)) * losses - costs
            minLoss = (args.eps * self.lambda_) + \
                  (self.lambda_ +args.k_) * ( torch.log(torch.mean(torch.exp( maxLoss - maxLoss.max() ))) + maxLoss.max() )  
            if(torch.rand(1)>0.9985):
                print("costs.mean:", costs.mean().item(), "minloss:", minLoss.item(), "max loss minimum:", torch.min(losses).item(), "max loss maximum: ", torch.max(losses).item(), success)
            return minLoss
    
    criterion = armor()
    
    optim = torch.optim.SGD(params, lr=args.lr_max, momentum=0.9, weight_decay=3.5e-3)

    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)

    if args.attack == 'free':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True
    elif args.attack == 'fgsm' and args.fgsm_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

    if args.attack == 'free':
        epochs = int(math.ceil(args.epochs / args.attack_iters))
    else:
        epochs = args.epochs

    if args.lr_schedule == 'superconverge':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]
    elif args.lr_schedule == 'piecewise':
        def lr_schedule(t):
            if t / args.epochs < 0.375: ## 75
                return args.lr_max
            elif t / args.epochs < 0.45: ## 90
                return args.lr_max / 10.
            elif t / args.epochs < 0.5: ## 100
                return args.lr_max / 100.
            else:
                return args.lr_max / 1000.
    elif args.lr_schedule == 'linear':
        lr_schedule = lambda t: np.interp([t], [0, args.epochs // 3, args.epochs * 2 // 3, args.epochs], [args.lr_max, args.lr_max, args.lr_max / 10, args.lr_max / 100])[0]
    elif args.lr_schedule == 'onedrop':
        def lr_schedule(t):
            if t < args.lr_drop_epoch:
                return args.lr_max
            else:
                return args.lr_one_drop
    elif args.lr_schedule == 'multipledecay':
        def lr_schedule(t):
            return args.lr_max - (t//(args.epochs//10))*(args.lr_max/10)
    elif args.lr_schedule == 'cosine':
        def lr_schedule(t):
            return args.lr_max * 0.5 * (1 + np.cos(t / args.epochs * np.pi))
    elif args.lr_schedule == 'cyclic':
        lr_schedule = lambda t: np.interp([t], [0, 0.4 * args.epochs, args.epochs], [0, args.lr_max, 0])[0]

    best_test_robust_acc = 0
    best_val_robust_acc = 0
    if args.resume:
        start_epoch = args.resume
        model.load_state_dict(torch.load(os.path.join(model_name, f'model_{start_epoch-1}.pth')))
        optim.load_state_dict(torch.load(os.path.join(model_name, f'opt_{start_epoch-1}.pth')))
        logger.info(f'Resuming at epoch {start_epoch}')

        if os.path.exists(os.path.join(model_name, f'model_best.pth')):
            best_test_robust_acc = torch.load(os.path.join(model_name, f'model_best.pth'))['test_robust_acc']
        if args.val:
            best_val_robust_acc = torch.load(os.path.join(model_name, f'model_val.pth'))['val_robust_acc']
    else:
        start_epoch = 0

    if args.eval:
        if not args.resume:
            logger.info("No model loaded to evaluate, specify with --resume FNAME")
            return
        logger.info("[Evaluation mode]")

    logger.info('Epoch \t TrainLoss \t TrainAc \t TrainRobustLoss \t TrainRobustAc \t TestingLoss_ \t TestAc \t TestRobustLoss \t TestRobustAc \t lambd')

    for epoch in range(start_epoch, epochs):
        start_time = time.time()
        train_loss = 0
        train_acc = 0
        train_robust_loss = 0
        train_robust_acc = 0
        train_n = 0
        for i, batch in enumerate(train_batches):
            if args.eval:
                break
            X, y = batch['input'], batch['target']
            lr = lr_schedule(epoch + (i + 1) / len(train_batches))
            optim.param_groups[0].update(lr=lr)

            if args.attack == 'pgd':
                # Random initialization
                delta = attack_udr(model, X, y, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm, lamda=criterion.lambda_)
                delta = delta.detach()
            # Standard training
            elif args.attack == 'none':
                delta = torch.zeros_like(X)
            X_adv = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)

            model.train()

            robust_output = model(normalize(X_adv))
            robust_loss = criterion(model=model, x_natural=X, x_adv = X_adv, y=y, tau=pgd_alpha)

            if args.l1:
                for name,param in model.named_parameters():
                    if 'bn' not in name and 'bias' not in name:
                        robust_loss += args.l1*param.abs().sum()

            optim.zero_grad()
            
            robust_loss.backward()
            
            optim.step()

            output = model(normalize(X))
            
            loss = F.cross_entropy(output, y)

            train_robust_loss += robust_loss.item() * y.size(0)
            train_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
        
        ## print ("lambda: ",criterion.lambda_)    
        train_time = time.time()

        model.eval()
        test_loss = 0
        test_acc = 0
        test_robust_loss = 0
        test_robust_acc = 0
        test_n = 0
        for i, batch in enumerate(test_batches):
            X, y = batch['input'], batch['target']

            # Random initialization
            if args.attack == 'none':
                delta = torch.zeros_like(X)
            else:
                delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
            delta = delta.detach()

            X_adv = torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)
            robust_output = model(normalize(X_adv))
            robust_loss = criterion(model=model, x_natural=X, x_adv = X_adv, y=y, tau=pgd_alpha)

            output = model(normalize(X))
            loss = F.cross_entropy(output, y)

            test_robust_loss += robust_loss.item() * y.size(0)
            test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            test_n += y.size(0)

        test_time = time.time()

        if args.val:
            val_loss = 0
            val_acc = 0
            val_robust_loss = 0
            val_robust_acc = 0
            val_n = 0
            for i, batch in enumerate(val_batches):
                X, y = batch['input'], batch['target']

                # Random initialization
                if args.attack == 'none':
                    delta = torch.zeros_like(X)
                else:
                    delta = attack_pgd(model, X, y, epsilon, pgd_alpha, args.attack_iters_test, args.restarts, args.norm, early_stop=args.eval)
                delta = delta.detach()

                robust_output = model(normalize(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)))
                robust_loss = criterion(robust_output, y)

                output = model(normalize(X))
                loss = criterion(output, y)

                val_robust_loss += robust_loss.item() * y.size(0)
                val_robust_acc += (robust_output.max(1)[1] == y).sum().item()
                val_loss += loss.item() * y.size(0)
                val_acc += (output.max(1)[1] == y).sum().item()
                val_n += y.size(0)

        if not args.eval:
            logger.info('%d \t %.0f \t %.4f \t \t %5.1f \t \t %.4f \t %.1f \t %.4f \t \t %.1f \t %.4f \t \t %.1f',
                epoch, train_loss/train_n, train_acc/train_n, train_robust_loss/train_n, train_robust_acc/train_n,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n, 
                criterion.lambda_)

            if args.val:
                logger.info('validation %.4f \t %.4f \t %.4f \t %.4f',
                    val_loss/val_n, val_acc/val_n, val_robust_loss/val_n, val_robust_acc/val_n)

                if val_robust_acc/val_n > best_val_robust_acc:
                    torch.save({
                            'state_dict':model.state_dict(),
                            'test_robust_acc':test_robust_acc/test_n,
                            'test_robust_loss':test_robust_loss/test_n,
                            'test_loss':test_loss/test_n,
                            'test_acc':test_acc/test_n,
                            'val_robust_acc':val_robust_acc/val_n,
                            'val_robust_loss':val_robust_loss/val_n,
                            'val_loss':val_loss/val_n,
                            'val_acc':val_acc/val_n,
                        }, os.path.join(model_name, f'model_val.pth'))
                    best_val_robust_acc = val_robust_acc/val_n

            # save checkpoint
            if (epoch+1) % args.chkpt_iters == 0 or epoch+1 == epochs:
                torch.save(model.state_dict(), os.path.join(model_name, f'model_{epoch}.pth'))
                torch.save(optim.state_dict(), os.path.join(model_name, f'opt_{epoch}.pth'))

            # save best
            if test_robust_acc/test_n > best_test_robust_acc:
                torch.save({
                        'state_dict':model.state_dict(),
                        'test_robust_acc':test_robust_acc/test_n,
                        'test_robust_loss':test_robust_loss/test_n,
                        'test_loss':test_loss/test_n,
                        'test_acc':test_acc/test_n,
                    }, os.path.join(model_name, f'model_best.pth'))
                best_test_robust_acc = test_robust_acc/test_n
        else:
            logger.info('%d \t %.1f \t \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t \t %.4f \t \t %.4f \t %.4f \t %.4f \t \t %.4f',
                epoch, train_time - start_time, test_time - train_time, -1,
                -1, -1, -1, -1,
                test_loss/test_n, test_acc/test_n, test_robust_loss/test_n, test_robust_acc/test_n)
            return


if __name__ == "__main__":
    main()
