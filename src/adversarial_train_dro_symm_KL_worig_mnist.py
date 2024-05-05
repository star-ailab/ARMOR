import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

##
from torch.autograd import Variable
from torcheval.metrics import MulticlassConfusionMatrix
import math
import random
import numpy


from utils.utils import load_parameters

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True) 
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
numpy.random.seed(0)
torch.backends.cudnn.deterministic = True

## Load parameters
parameters = load_parameters("../src/parameters_dro_symm_KL_worig.ini")
L = float(parameters["hyperparam"]["ot_norm_weight"]) 
q = float(parameters["hyperparam"]["ot_norm_power"])
norm_ = float(parameters["hyperparam"]["ot_norm"])
eps = float(parameters["hyperparam"]["eps"])
beta_ = float(parameters["hyperparam"]["beta"])
t_ = float(parameters["hyperparam"]["t"])

softpl = nn.Softplus(beta = beta_) ## default beta is 1. Lower than 1 makes the function increase faster
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

## Not used.
def pgd_linf_OT_NoProjection(model, X, y, lambda_, epsilon=0.1, step_alpha=0.1, num_iter=20, randomize=False):
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)
        
    for t in range(num_iter):
        loss = (lambda_ * loss_fct(model(X + delta), y)) - ( L * (torch.linalg.norm(torch.flatten(X, start_dim=1) - torch.flatten((X + delta), start_dim=1), dim = 1, ord=norm_))**q )
        loss.mean().backward()
        delta.data = (delta + step_alpha*delta.grad.detach().sign())
        delta.grad.zero_()
    return delta.detach()

def pgd_linf_OT_NoProjection2(model, x, y, lambda_, epsilon=0.1, step_alpha=0.01, num_iter=20):
    # some book-keeping
    if next(model.parameters()).is_cuda:
        x = x.cuda()
        y = y.cuda()
    y = Variable(y)
    # compute natural loss
    loss_natural = loss_fct(model(Variable(x)), y).data
    # initialize starting point
    ## x_next = torch.tensor(x, requires_grad=True)
    x_next = x.clone().detach().requires_grad_(True)
    losses = []

    for t in range(num_iter):
        # forward pass
        x_var = Variable(x_next, requires_grad=True)
        y_model = model(x_var)
        loss = ((1.0/lambda_) * loss_fct(model(x_var), y)) - ( L * (torch.linalg.norm(torch.flatten(x, start_dim=1) - torch.flatten((x_var), start_dim=1), dim = 1, ord=norm_))**q )
        ## print("LOSS:", loss.mean())
        losses.append(loss.cpu().data.numpy()[0])
        # compute gradient
        grad_vars = torch.autograd.grad(loss.mean(), x_var)
        
        x_next = x_next + step_alpha * torch.sign(grad_vars[0].data)
        
    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data
    
    ### print ("adversarial Loss: ", loss_adv.mean())

    losses.append(loss_adv)

    return x_next    
        
## Our Outer losses
class min_loss_KL_symmetric (nn.Module):
    def __init__(self):
        super(min_loss_KL_symmetric, self).__init__()
        self.lambda_ = nn.Parameter(torch.tensor(1.0, requires_grad=True))
    
    def forward(self, x_adv_model, y, x_adv, x):
        maxLoss = (1.0/self.lambda_) *(loss_fct(x_adv_model, y.cuda())) - L *  (torch.linalg.norm(torch.flatten(x.cuda(), start_dim=1) - torch.flatten(x_adv.cuda(), start_dim=1), dim = 1, ord=norm_))**q
        minLoss = (eps * self.lambda_) + \
                  (self.lambda_) * ( torch.log(torch.mean(torch.exp( maxLoss - maxLoss.max() ))) + maxLoss.max() ) 
        
        # print ("maxLoss: ", maxLoss.mean())
        # print ("minLoss: ", minLoss)
        # print ("raw lambda_: ", self.lambda_.data)
        if (math.isnan(maxLoss.mean()) or math.isnan(minLoss)):
            print("***************** NaN Loss *********** Training is invalid")
        
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


def epoch_adversarial(loader, model, attack, opt=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    ##
    ys = []
    yps = []
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            optimizer_loss_params.zero_grad()
            
            loss.backward()
            
            opt.step()
            optimizer_loss_params.step()
        
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

def epoch_adversarial_dro(loader, model, attack, lambda_, opt=None, optimizer_loss_params=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        yp_orig = model(X)
        
        x_adv = attack(model, X, y, lambda_, **kwargs)
        yp = model(x_adv)
        
        ##loss = criterion(yp, y, X+delta, X)        
        loss_orig = loss_fct(model(X), y).mean()
        loss_adv = criterion(yp, y, x_adv, X)
        loss = t_ * loss_orig + (1-t_) * loss_adv
        if opt:
            opt.zero_grad()
            optimizer_loss_params.zero_grad()
            
            loss.backward()
            
            opt.step()
            optimizer_loss_params.step()
            ## failsafe for lambda in case it is negative. Lambda should not be negative unldess the learning rate is too large.
            if (lambda_<0):
                print ("*****lambda was negative. Failsafe mechanism was invoked")
                criterion.lambda_ = nn.Parameter(softpl(lambda_))
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        
    ## print("lambda: ", criterion.lambda_.data.item(), "loss: ", loss.data.item() )
    
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

def epoch_adversarial_dro_eval(loader, model, attack, lambda_, opt=None, optimizer_loss_params=None, **kwargs):
    total_loss, total_err = 0.,0.
    ##
    ys = []
    yps = []
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        x_adv = attack(model, X, y, lambda_, **kwargs)
        yp = model(x_adv)
        
        ##loss = criterion(yp, y, X+delta, X)        
        loss = criterion(yp, y, x_adv, X)
        if opt:
            opt.zero_grad()
            optimizer_loss_params.zero_grad()
            
            loss.backward()
            
            opt.step()
            optimizer_loss_params.step()
        
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

criterion = min_loss_KL_symmetric()
opt = optim.SGD(model_cnn_robust_dro_KL_symm_worig.parameters(), lr=1e-1)
optimizer_loss_params = optim.Adam(list(criterion.parameters()), lr=float(parameters["hyperparam"]["loss_learning_rate"]))

## print hyperparameters
print ( "criterion: ", type(criterion), "dro_epsilon: ", float(parameters["hyperparam"]["eps"]), "L: ", float(parameters["hyperparam"]["ot_norm_weight"]),\
                        "q: ", float(parameters["hyperparam"]["ot_norm_power"]), "norm: ", float(parameters["hyperparam"]["ot_norm"]),\
                        "beta: ", float(parameters["hyperparam"]["beta"]), "alpha: ", float(parameters["hyperparam"]["alpha"]),\
                        "t: ", float(parameters["hyperparam"]["t"]), "lr_lambda_nu: ",  float(parameters["hyperparam"]["loss_learning_rate"]), \
                        )
print("train_err,       adv_acc,       adv_FPR,        adv_FNR,       test_acc,       test_FPR,       test_FNR")

for t in range(10):
    train_err, train_loss = epoch_adversarial_dro(train_loader, model_cnn_robust_dro_KL_symm_worig, pgd_linf_OT_NoProjection2, criterion.lambda_, opt, optimizer_loss_params)
    test_err, test_loss, test_FPR, test_FNR = epoch(test_loader, model_cnn_robust_dro_KL_symm_worig)
    adv_err, adv_loss, adv_FPR, adv_FNR = epoch_adversarial_dro_eval(test_loader, model_cnn_robust_dro_KL_symm_worig, pgd_linf_OT_NoProjection2, criterion.lambda_)
    if t == 4:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-2
    print(*("{:.6f}".format(i) for i in (train_err, 1.0-adv_err, 100.0*adv_FPR, 100.0*adv_FNR, 1.0-test_err, 100.0* test_FPR, 100.0*test_FNR)), sep="\t")

torch.save(model_cnn_robust_dro_KL_symm_worig.state_dict(), "model_cnn_robust_dro_KL_symm_worig.pt")


##################### Evaluation

fgsm_err , _, fgsm_FPR, fgsm_FNR = epoch_adversarial(test_loader, model_cnn_robust_dro_KL_symm_worig, fgsm)

print("fgsm acc", 1.0-fgsm_err, "fgsm FPR: ", 100.0*fgsm_FPR, "fgsm FNR: ", 100.0*fgsm_FNR)

pgd_err , _, pgd_FPR, pgd_FNR = epoch_adversarial(test_loader, model_cnn_robust_dro_KL_symm_worig, pgd_linf, num_iter=40)

print("pgd_linf, 40 iter acc: ", 1.0-pgd_err, "pgd FPR: ", 100.0*pgd_FPR, "pgd FNR: ", 100.0*pgd_FNR)




