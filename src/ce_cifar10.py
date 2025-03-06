from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import load_parameters


from resnet import *
from mart_persample import mart_loss
from trades_persample import trades_loss
import numpy as np
import time

import scipy.optimize as opt

from autoattack import AutoAttack
from pickle import FALSE

parser = argparse.ArgumentParser(description='CIFAR ARMOR+TRADES/MART Defense')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N', help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=3.5e-3, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--epsilon', default=8./255., help='perturbation')
parser.add_argument('--num-steps', default=10, help='perturb number of steps')
parser.add_argument('--step-size', default=2./255., help='perturb step size')
parser.add_argument('--beta', default=19.0, help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--model', default='resnet', help='directory of model for saving checkpoint')
parser.add_argument('--pgd_interval', default=40, help='PGD attack interval')
parser.add_argument('--autoattack_interval', default=40, help='autoattack attack interval')
parser.add_argument('--save-freq', '-s', default=200, type=int, metavar='N', help='save frequency')


## armor params
parser.add_argument('--eps', default=0.02, help='armor epsilon')
parser.add_argument('--lambd', default=1.0, help='armor initial lambda')
parser.add_argument('--t', default=0.5, help='natural vs. adv. loss (between 0 and 1)')
parser.add_argument('--k_', default=0.05, help='lower-boud ARMOR loss via lregularization')
parser.add_argument('--verbose', action='store_true', default=False, help='print armor parameters after each minibatch')

parser.add_argument('--num-steps-eval', default=200, help='evaluation - perturbation number of steps')

args = parser.parse_args()
print(args)

# settings
model_dir = args.model
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    

    
log_dir = './log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    
use_cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
torch.backends.cudnn.benchmark = True

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = torchvision.datasets.CIFAR10(root='../data_attack/', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='../data_attack/', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

##For auto attack, following Bui et al.
random_sampler = torch.utils.data.RandomSampler(testset, num_samples=1000)
test_loader_1000 = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0, sampler=random_sampler)

## armor
def lambda_objective_derivative(epsilon,kappa,lambda_,loss_array):
    
    max_loss=np.max(loss_array)
    return epsilon+np.log(np.average(np.exp((loss_array-max_loss)/(lambda_+kappa))))+max_loss/(lambda_+kappa)-(lambda_+kappa)**(-1)*np.average(loss_array*np.exp((loss_array-max_loss)/(lambda_+kappa)))/np.average(np.exp((loss_array-max_loss)/(lambda_+kappa)))


def optimize_lambda(epsilon,kappa,loss_array,lambda_b=10,lambda_max=100):
    if lambda_objective_derivative(epsilon,kappa,0,loss_array)>=0:
        return 0, True
    else:
        while lambda_objective_derivative(epsilon,kappa,lambda_b,loss_array)<=0:
            lambda_b=2*lambda_b
            if lambda_b>=lambda_max:
                return lambda_max, False
            
        lambda_star=opt.bisect(lambda z: lambda_objective_derivative(epsilon,kappa,z,loss_array),0,lambda_b)

        return lambda_star, True

class armor (nn.Module):
    def __init__(self):
        super(armor, self).__init__()
        self.lambda_ = args.lambd
    
    def forward(self, model, x_natural, y, optimizer):
        ## pdist = torch.nn.PairwiseDistance(p=norm_)
        losses = (trades_loss(model=model, x_natural=x_natural, y=y.to(device), optimizer=optimizer, device=device, beta=args.beta))
        self.lambda_, success = optimize_lambda(args.eps, args.k_, losses.detach().cpu().numpy())
        maxLoss = (1.0/(self.lambda_+ args.k_)) * losses
        minLoss = (args.eps * self.lambda_) + \
                  (self.lambda_ +args.k_) * ( torch.log(torch.mean(torch.exp( maxLoss - maxLoss.max() ))) + maxLoss.max() )  
        if(torch.rand(1)>0.998):
            print("minloss:", minLoss.item(), "max loss minimum:", torch.min(losses).item(), "max loss maximum: ", torch.max(losses).item(), success)
        return minLoss

criterion = armor()
loss_fct = nn.CrossEntropyLoss(reduction='none')

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss (trades)
        ##loss_robust = criterion(model=model, x_natural=data, y=target, optimizer=optimizer)
        
        loss = loss_fct(model(data), target).mean()
        
        ##loss = args.t * loss_nat + (1- args.t) * loss_robust
        
        
        loss.backward()
        # gradient norm bound, following previous work setting
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            
            if args.verbose:
                print ("lambda: ",criterion.lambda_)
    print ("lambda: ",criterion.lambda_)
def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=args.epsilon,
                  num_steps=args.num_steps_eval,
                  step_size=2./255.):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)

    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    return err, err_pgd

def test_aa(model, data_loader, device): 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    return accuracy

def eval_adv_test_whitebox(model, device, test_loader, epoch, attacks):

    model.eval()
    robust_err_total = 0
    natural_err_total = 0

    pgd_robust_acc = 0.
    if ('pgd' in attacks and epoch % args.pgd_interval == 0) or ('pgd' in attacks and epoch in [150]):
        torch.manual_seed(args.seed)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = _pgd_whitebox(model, X, y)
            robust_err_total += err_robust
            natural_err_total += err_natural
        
        print('natural_acc: ', (1 - natural_err_total / len(test_loader.dataset)).item() )
        pgd_robust_acc = (1.- robust_err_total / len(test_loader.dataset)).item()
        print('pgd_robust_acc: ', (1.- robust_err_total / len(test_loader.dataset)).item() )
    
    ## Autoattack:
    if ('aa' in attacks and epoch % args.autoattack_interval == 0):
        torch.manual_seed(args.seed)
        for X_,y_ in test_loader_1000:
            X_,y_ = X_.to(device), y_.to(device)
            break
        adversary = AutoAttack(model, norm='Linf', eps=8./255., verbose= False, device="cuda")
        adv_complete = adversary.run_standard_evaluation(X_, y_,bs=100)
        
        aa_data = torch.utils.data.TensorDataset(adv_complete, y_)
        aa_loader = torch.utils.data.DataLoader(aa_data, batch_size=1000, shuffle=False, num_workers=0)
        adv_acc = test_aa(model, aa_loader, device)
        print("Auto Attack Accuracy: ", adv_acc)
    
    return 1 - natural_err_total / len(test_loader.dataset), 1- robust_err_total / len(test_loader.dataset)


def main():

    model = ResNet18(num_classes=10).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    natural_acc = []
    robust_acc = []
    
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        
        start_time = time.time()

        # adversarial training
        train(args, model, device, train_loader, optimizer, epoch)


        print('================================================================')
        natural_err_total, robust_err_total = eval_adv_test_whitebox(model, device, test_loader, epoch, attacks=['aa', 'pgd'])
        
        ##print('time:', time.time()-start_time)
        
        natural_acc.append(natural_err_total)
        robust_acc.append(robust_err_total)
        print('================================================================')
        
        file_name = os.path.join(log_dir, 'train_stats.npy')

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'model-res-epoch{}.pt'.format(epoch)))
            torch.save(optimizer.state_dict(),
                       os.path.join(model_dir, 'opt-res-checkpoint_epoch{}.tar'.format(epoch)))


if __name__ == '__main__':
    main()

