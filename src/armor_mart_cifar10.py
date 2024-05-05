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
from mart_persample_consistent import mart_loss
import numpy as np
import time

from autoattack import AutoAttack
from pickle import FALSE

parser = argparse.ArgumentParser(description='PyTorch CIFAR MART Defense')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=3.5e-3,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8./255.,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2./255.,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0,
                    help='weight before kl (misclassified examples)')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model', default='resnet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=50, type=int, metavar='N',
                    help='save frequency')


## armor params
parser.add_argument('--loss_learning_rate', default=0.000006, help='armor lambda and nu learning rate')
parser.add_argument('--eps', default=0.5, help='armor epsilon')
parser.add_argument('--alph', default=10.0, help='armor alpha')
parser.add_argument('--lambd', default=1.0, help='armor initial lambda')
parser.add_argument('--nu', default=1.0, help='armor initial nu')

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

##For auto attack, as in Bui et al.
random_sampler = torch.utils.data.RandomSampler(testset, num_samples=1000)
test_loader_1000 = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=0, sampler=random_sampler)

## armor
def f_star(y, alph):
    return (1.0/alph) * ((alph-1)**(alph/(alph-1))) * ( (torch.relu(y))**(alph/(alph-1)) ) + 1.0/(alph*(alph-1))

class min_loss_fdiv_symmetric (nn.Module):
    def __init__(self):
        super(min_loss_fdiv_symmetric, self).__init__()
        self.lambda_ = nn.Parameter(torch.tensor(args.lambd, requires_grad=True))
        self.nu = nn.Parameter(torch.tensor(args.nu, requires_grad=True))
    
    def forward(self, model, x_natural, y, optimizer):
        maxLoss = (1.0/self.lambda_)*(mart_loss(model, x_natural, y.to(device), optimizer, device, beta=args.beta))
        minLoss = (args.eps * (self.lambda_)) + self.nu +\
                  (self.lambda_) * ( torch.mean(f_star(maxLoss - ((1.0/self.lambda_)*self.nu), args.alph))  ) 
        if(torch.rand(1)>0.985):
            print("minloss:", minLoss.item(), "max loss minimum:", torch.min(maxLoss - ((1.0/self.lambda_)*self.nu)).item(), "max loss maximum: ", torch.max(maxLoss - ((1.0/self.lambda_)*self.nu)).item())
        return minLoss

criterion = min_loss_fdiv_symmetric()


def train(args, model, device, train_loader, optimizer, optimizer_loss_params, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        optimizer_loss_params.zero_grad()

        # calculate robust loss (mart)
        loss = criterion(model=model,
                           x_natural=data,
                           y=target,
                           optimizer=optimizer)
        
        
        loss.backward()
        optimizer.step()
        optimizer_loss_params.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            
            print ("lambda: ",criterion.lambda_.item(), "nu: ", criterion.nu.item())

def adjust_learning_rate(optimizer, optimizer_loss_params, epoch):
    """decrease the learning rate"""
    lr = args.lr
    lr_params = args.loss_learning_rate
    if epoch >= 100:
        lr = args.lr * 0.001
        lr_params = args.loss_learning_rate * 0.001
    elif epoch >= 90:
        lr = args.lr * 0.01
        lr_params = args.loss_learning_rate * 0.01
    elif epoch >= 75:
        lr = args.lr * 0.1
        lr_params = args.loss_learning_rate * 0.1
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

    if ('pgd' in attacks and epoch % 1 == 0):
        torch.manual_seed(args.seed) 
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            err_natural, err_robust = _pgd_whitebox(model, X, y)
            robust_err_total += err_robust
            natural_err_total += err_natural
    
    ## Autoattack:
    if ('aa' in attacks and epoch % 1 == 0):
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
    
    print('natural_acc: ', 1 - natural_err_total / len(test_loader.dataset))
    print('robust_acc: ', 1- robust_err_total / len(test_loader.dataset))
    return 1 - natural_err_total / len(test_loader.dataset), 1- robust_err_total / len(test_loader.dataset)


def main():

    model = ResNet18().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_loss_params = optim.AdamW(list(criterion.parameters()), lr=args.loss_learning_rate)
    
    natural_acc = []
    robust_acc = []
    
    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, optimizer_loss_params, epoch)
        
        start_time = time.time()

        # adversarial training
        train(args, model, device, train_loader, optimizer, optimizer_loss_params, epoch)


        print('================================================================')
        natural_err_total, robust_err_total = eval_adv_test_whitebox(model, device, test_loader, epoch, attacks=['aa', 'pgd'])
        
        print('using time:', time.time()-start_time)
        
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

