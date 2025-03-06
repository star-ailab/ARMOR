import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse

##
from torch.autograd import Variable
from torcheval.metrics import MulticlassConfusionMatrix
import math
import random
import numpy as np

import scipy.optimize as opt


mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True) 
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description='MNIST ARMOR Defense')
parser.add_argument('--rho', default=-0.35, help='armor initial rho')
parser.add_argument('--lambd', default=1.0, help='armor initial lambda')

parser.add_argument('--eps', default=0.1, help='armor epsilon')
parser.add_argument('--t_', default=0., help='natural vs. adv. loss (between 0 and 1)')
parser.add_argument('--alph', default=1.25, help='armor alpha')
parser.add_argument('--k_', default=0.1, help='lower-boud ARMOR loss via regularization')

args = parser.parse_args()
print(args)

loss_fct = nn.CrossEntropyLoss(reduce=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    


############# Attacks (Inner Maximizers)

def fgsm(model, X, y, epsilon=0.1):
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.1, alpha=0.01, num_iter=20, randomize=False):
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()




##armor

def Renyi_lambda_rho_objective(lambda_,rho_,epsilon,kappa,alpha,loss_array):
    val=lambda_*epsilon+rho_-(lambda_+kappa)*alpha**(-1)*(np.log(alpha)+1.)+alpha**(-1)*(lambda_+kappa)*np.log(lambda_+kappa)
    val=val-(lambda_+kappa)/(alpha-1.)*np.log(np.average((rho_-loss_array)**((alpha-1)/alpha)))

    return val

def Renyi_lambda_rho_objective_grad(lambda_,rho_,epsilon,kappa,alpha,loss_array):
    
    partial_lambda=epsilon-alpha**(-1)*(np.log(alpha)+1.)+alpha**(-1)*(np.log(lambda_+kappa)+1)-1./(alpha-1.)*np.log(np.average((rho_-loss_array)**((alpha-1)/alpha)))
    partial_rho=1.-((lambda_+kappa)/alpha)*np.average((rho_-loss_array)**(-1./alpha))/np.average((rho_-loss_array)**((alpha-1)/alpha))
    return [partial_lambda,partial_rho]

def optimize_Renyi_lambda_rho(epsilon,kappa,alpha,loss_array,lambda_0=0.):
        #x=[lambda_,rho_]
        obj_fun = lambda x:  Renyi_lambda_rho_objective(x[0],x[1],epsilon,kappa,alpha,loss_array)
        grad_fun= lambda x: Renyi_lambda_rho_objective_grad(x[0],x[1],epsilon,kappa,alpha,loss_array)

        lambda_min=0.
        safety_tol=1e-6
        rho_min=np.max(loss_array)+ safety_tol
        bnds= ((lambda_min, None), (rho_min, None))
        tol=1e-7
        res = opt.minimize(obj_fun, jac=grad_fun, x0=[lambda_0,rho_min+1.],  bounds=bnds, tol=tol )
        
        lambda_star=res.x[0]
        rho_star=res.x[1]
        
        ##print(res)

        return np.maximum(lambda_star,0), rho_star, res.success

class armor (nn.Module):
    def __init__(self):
        super(armor, self).__init__()
        self.lambda_ = args.lambd
        self.rho_ = args.rho
    
    def forward(self, x_adv_model, y):
        losses = loss_fct(x_adv_model, y)
        self.lambda_, self.rho_, success = optimize_Renyi_lambda_rho(args.eps, args.k_, args.alph, losses.detach().cpu().numpy())
        
        val=self.lambda_*args.eps+self.rho_-(self.lambda_+args.k_)*args.alph**(-1)*(np.log(args.alph)+1.)+args.alph**(-1)*(self.lambda_+args.k_)*\
        np.log(self.lambda_+args.k_)
        minLoss = val-(self.lambda_+args.k_)/(args.alph-1.)*torch.log(torch.mean((self.rho_-losses)**((args.alph-1)/args.alph)))
        
        if(torch.rand(1)>0.99998):
            print ("minLoss: ", minLoss.item(), "success: ", success)
        
        return minLoss


def epoch(loader, model, opt=None):
    """Standard training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    ##
    ys = []
    yps = []
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        ys.append(y)
        yps.append(yp.max(dim=1)[1])
        
    ## compute metrics (micro averaging: MNIST is balanced so we can use micro-averaging and give equal weight to each classified sample)
    ys = torch.cat(ys)
    yps = torch.cat(yps)
    metric = MulticlassConfusionMatrix(10)
    metric.update(yps.cpu(), ys.cpu())
    cf = metric.compute()
    FP = cf.sum(dim=0) - torch.diag(cf) 
    FN = cf.sum(dim=1) - torch.diag(cf)
    TP = torch.diag(cf)
    TN = cf.sum() - (FP + FN + TP)
    
    FPR = torch.mean( FP/(FP+TN) )
    FNR = torch.mean( FN/(TP+FN) )
        
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), FPR, FNR


def epoch_adversarial_dro(loader, model, attack, opt=None):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        yp_orig = model(X)
        
        delta = attack(model, X, y)
        x_adv = X+delta
        yp = model(x_adv)
        
        loss_orig = loss_fct(model(X), y).mean()
        loss_adv = criterion(yp, y)
        loss = args.t_* loss_orig + (1-args.t_) * loss_adv
        if opt:
            opt.zero_grad()
            
            loss.backward()
            
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    ##print("lambda: ", criterion.lambda_.item(), "loss_orig: ", loss_orig.item(), "loss_adv: ", loss_adv.item() )
    
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def epoch_adversarial_pgd_eval(loader, model, attack, num_iter=40):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    ##
    ys = []
    yps = []
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, num_iter=num_iter)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
                
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        ys.append(y)
        yps.append(yp.max(dim=1)[1])
        
    ## print("lambda: ", criterion.lambda_.data.item(), "loss: ", loss.data.item() )
    
    ## compute metrics (micro averaging: MNIST is balanced so we can use micro-averaging and give equal weight to each classified sample)
    ys = torch.cat(ys)
    yps = torch.cat(yps)
    metric = MulticlassConfusionMatrix(10)
    metric.update(yps.cpu(), ys.cpu())
    cf = metric.compute()
    FP = cf.sum(dim=0) - torch.diag(cf) 
    FN = cf.sum(dim=1) - torch.diag(cf)
    TP = torch.diag(cf)
    TN = cf.sum() - (FP + FN + TP)
    
    FPR = torch.mean( FP/(FP+TN) )
    FNR = torch.mean( FN/(TP+FN) )
    
    
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), FPR, FNR

def epoch_adversarial_fgsm_eval(loader, model, attack):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    ##
    ys = []
    yps = []
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        ys.append(y)
        yps.append(yp.max(dim=1)[1])
        
    ## print("lambda: ", criterion.lambda_.data.item(), "loss: ", loss.data.item() )
    
    ## compute metrics (micro averaging: MNIST is balanced so we can use micro-averaging and give equal weight to each classified sample)
    ys = torch.cat(ys)
    yps = torch.cat(yps)
    metric = MulticlassConfusionMatrix(10)
    metric.update(yps.cpu(), ys.cpu())
    cf = metric.compute()
    FP = cf.sum(dim=0) - torch.diag(cf) 
    FN = cf.sum(dim=1) - torch.diag(cf)
    TP = torch.diag(cf)
    TN = cf.sum() - (FP + FN + TP)
    
    FPR = torch.mean( FP/(FP+TN) )
    FNR = torch.mean( FN/(TP+FN) )
    
    
    return total_err / len(loader.dataset), total_loss / len(loader.dataset), FPR, FNR

#########################################################################
############### Adversarial Training with OT-Regularized Inner maximizer

print ("Adversarial Training with OT-Regularized Inner Maximizer (Ours)")

model_cnn_robust_dro_KL_symm_worig = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                                 Flatten(),
                                 nn.Linear(7*7*64, 100), nn.ReLU(),
                                 nn.Linear(100, 10)).to(device)

criterion = armor()
optimizer = optim.SGD(model_cnn_robust_dro_KL_symm_worig.parameters(), lr=1e-1)

print("epoch,     train_err,       adv_acc,       adv_FPR,        adv_FNR,       test_acc,       test_FPR,       test_FNR")

for t in range(10):
    train_err, train_loss = epoch_adversarial_dro(train_loader, model_cnn_robust_dro_KL_symm_worig, pgd_linf, optimizer)
    test_err, test_loss, test_FPR, test_FNR = epoch(test_loader, model_cnn_robust_dro_KL_symm_worig)
    adv_err, adv_loss, adv_FPR, adv_FNR = epoch_adversarial_pgd_eval(test_loader, model_cnn_robust_dro_KL_symm_worig, pgd_linf, num_iter=10)
    if t == 2:
        for param_group in optimizer.param_groups:
            param_group["lr"] = 1e-2
    print(t, *("{:.6f}".format(i) for i in (train_err, 1.0-adv_err, 100.0*adv_FPR, 100.0*adv_FNR, 1.0-test_err, 100.0* test_FPR, 100.0*test_FNR)), sep="\t")

##torch.save(model_cnn_robust_dro_KL_symm_worig.state_dict(), "model_cnn_robust_dro_KL_symm_worig.pt")


##################### Evaluation

fgsm_err , _, fgsm_FPR, fgsm_FNR = epoch_adversarial_fgsm_eval(test_loader, model_cnn_robust_dro_KL_symm_worig, fgsm)

print("fgsm acc", 1.0-fgsm_err, "fgsm FPR: ", 100.0*fgsm_FPR, "fgsm FNR: ", 100.0*fgsm_FNR)

pgd_err , _, pgd_FPR, pgd_FNR = epoch_adversarial_pgd_eval(test_loader, model_cnn_robust_dro_KL_symm_worig, pgd_linf, num_iter=40)

print("pgd_linf, 40 iter acc: ", 1.0-pgd_err, "pgd FPR: ", 100.0*pgd_FPR, "pgd FNR: ", 100.0*pgd_FNR)




