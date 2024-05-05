'''
Created on Jun 30, 2023
Implemented based on https://adversarial-ml-tutorial.org/adversarial_training/
@author: eb

'''
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.utils import load_parameters

##
from torch.autograd import Variable
from torcheval.metrics import MulticlassConfusionMatrix
import random
import numpy

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True, num_workers=0)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False, num_workers=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
numpy.random.seed(0)
torch.backends.cudnn.deterministic = True

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)    

model_cnn = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                          nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                          Flatten(),
                          nn.Linear(7*7*64, 100), nn.ReLU(),
                          nn.Linear(100, 10)).to(device)

loss_fct = nn.CrossEntropyLoss(reduce=False)

############# Attacks (Inner Maximizers)

def fgsm(model, X, y, epsilon=0.1):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()

def pgd_linf(model, X, y, epsilon=0.3, alpha=0.01, num_iter=40, randomize=False):
    """ Construct FGSM adversarial examples on the examples X"""
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

##
def pgd_linf_OT_NoProjection2(model, x, y, lambda_, epsilon=0.1, step_alpha=0.01, num_iter=20):
    """ Construct FGSM adversarial examples on the examples X"""
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
        loss = (lambda_ * loss_fct(model(x_var), y)) - ( 0.1 * (torch.linalg.norm(torch.flatten(x, start_dim=1) - torch.flatten((x_var), start_dim=1), dim = 1, ord=2.0))**1.0 )
        ## print("LOSS:", loss.mean())
        losses.append(loss.cpu().data.numpy()[0])
        # compute gradient
        grad_vars = torch.autograd.grad(loss.sum(), x_var)
        
        # print(grad_vars[0].sum())
        # find the next sample
        ## clamp the gradients to unit ball
        
        x_next = x_next + step_alpha * torch.sign(grad_vars[0].data)
        
        ## Clamping does not help getting better results on other atatcks        
        ##x_next = x_next + step_alpha * torch.clamp(grad_vars[0].data, -0.8, 0.8)
        
        # projection Note: we do not have projection
        ## x_next = clip_tensor(x_next)
    # compute adversarial loss
    loss_adv = loss_fct(model(Variable(x_next)), y).data
    
    ### print ("adversarial Loss: ", loss_adv.mean())

    losses.append(loss_adv)

    return x_next    

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
    count = 0
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        delta = attack(model, X, y, **kwargs)
        yp = model(X+delta)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        ys.append(y)
        yps.append(yp.max(dim=1)[1])
        count = count +1
    print ('****', count)
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

def epoch_adversarial_dro_eval(loader, model, attack, lambda_, opt=None, optimizer_loss_params=None, **kwargs):
    """Adversarial training/evaluation epoch over the dataset"""
    total_loss, total_err = 0.,0.
    ##
    ys = []
    yps = []
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        
        ##delta = attack(model, X, y, **kwargs)
        ##yp = model(X+delta)
        ##
        x_adv = attack(model, X, y, lambda_, **kwargs)
        yp = model(x_adv)
        
        ##loss = criterion(yp, y, X+delta, X)        
        ##loss = criterion(yp, y, x_adv, X)
        # if opt:
        #     opt.zero_grad()
        #     optimizer_loss_params.zero_grad()
        #
        #     loss.backward()
        #
        #     opt.step()
        #     optimizer_loss_params.step()
        
        total_err += (yp.max(dim=1)[1] != y).sum().item()
        ##total_loss += loss.item() * X.shape[0]
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
    
    ## Sensitivity, hit rate, recall, or true positive rate
    ##TPR = TP/(TP+FN)
    ## Specificity or true negative rate
    ##TNR = TN/(TN+FP) 
    ## Precision or positive predictive value
    ##PPV = TP/(TP+FP)
    ## Negative predictive value
    ##NPV = TN/(TN+FN)
    ## Fall out or false positive rate
    FPR = torch.mean( FP/(FP+TN) )
    # False negative rate
    FNR = torch.mean( FN/(TP+FN) )
    # False discovery rate
    ##FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ##ACC = (TP+TN)/(TP+FP+FN+TN)
    
    
    return total_err / len(loader.dataset), FPR, FNR

        
print ("Non-adversarial Training, Attack by PGD")

# opt = optim.SGD(model_cnn.parameters(), lr=1e-1)
# for t in range(10):
#     train_err, train_loss, train_FPR, train_FNR = epoch(train_loader, model_cnn, opt)
#     test_err, test_loss, test_FPR, test_FNR = epoch(test_loader, model_cnn)
#     adv_err, adv_loss, adv_FPR, adv_FNR = epoch_adversarial(test_loader, model_cnn, pgd_linf)
#     if t == 4:
#         for param_group in opt.param_groups:
#             param_group["lr"] = 1e-2
#     print(*("{:.6f}".format(i) for i in (train_err, 1.0-adv_err, 100.0*adv_FPR, 100.0*adv_FNR, 1.0-test_err, 100.0* test_FPR, 100.0*test_FNR)), sep="\t")
# torch.save(model_cnn.state_dict(), "model_cnn.pt")
#
# model_cnn.load_state_dict(torch.load("model_cnn.pt"))
#
# print("FGSM ", epoch_adversarial(test_loader, model_cnn, fgsm))
# print("PGD, 40 iter: ", epoch_adversarial(test_loader, model_cnn, pgd_linf, num_iter=40))
# print("pgd_linf_OT, 40 iter: ", epoch_adversarial_dro_eval(test_loader, model_cnn, pgd_linf_OT_NoProjection2, 10.0, num_iter=40))

########## Adversarial Training with FGSM

print ("Adversarial Training with FGSM")

# model_cnn_robust_fgsm = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
#                                  nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
#                                  nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
#                                  nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
#                                  Flatten(),
#                                  nn.Linear(7*7*64, 100), nn.ReLU(),
#                                  nn.Linear(100, 10)).to(device)
#
# opt = optim.SGD(model_cnn_robust_fgsm.parameters(), lr=1e-1)
# for t in range(10):
#     train_err, train_loss, _, _ = epoch_adversarial(train_loader, model_cnn_robust_fgsm, fgsm, opt)
#     test_err, test_loss, test_FPR, test_FNR = epoch(test_loader, model_cnn_robust_fgsm)
#     adv_err, adv_loss, adv_FPR, adv_FNR = epoch_adversarial(test_loader, model_cnn_robust_fgsm, fgsm)
#     if t == 4:
#         for param_group in opt.param_groups:
#             param_group["lr"] = 1e-2
#     print(*("{:.6f}".format(i) for i in (train_err, 1.0-adv_err, 100.0*adv_FPR, 100.0*adv_FNR, 1.0-test_err, 100.0* test_FPR, 100.0*test_FNR)), sep="\t")
# torch.save(model_cnn_robust_fgsm.state_dict(), "model_cnn_robust_fgsm.pt")
#
# print("FGSM ", epoch_adversarial(test_loader, model_cnn_robust_fgsm, fgsm))
# print("PGD, 40 iter: ", epoch_adversarial(test_loader, model_cnn_robust_fgsm, pgd_linf, num_iter=40))
# print("pgd_linf_OT, 40 iter: ", epoch_adversarial_dro_eval(test_loader, model_cnn_robust_fgsm, pgd_linf_OT_NoProjection2, 10.0, num_iter=40))

########################################
########## Adversarial Training with PGD

print ("Adversarial Training with PGD")

model_cnn_robust = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                                 Flatten(),
                                 nn.Linear(7*7*64, 100), nn.ReLU(),
                                 nn.Linear(100, 10)).to(device)

model_cnn_robust = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
                                 nn.MaxPool2d(2),
                                 Flatten(),
                                 nn.Linear(256, 200), nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(200, 200), nn.ReLU(),
                                 nn.Linear(200, 10)).to(device)

opt = optim.SGD(model_cnn_robust.parameters(), lr=1e-2, momentum=0.9)
for t in range(10):
    train_err, train_loss, _, _ = epoch_adversarial(train_loader, model_cnn_robust, pgd_linf, opt)
    test_err, test_loss, test_FPR , test_FNR = epoch(test_loader, model_cnn_robust)
    adv_err, adv_loss, adv_FPR, adv_FNR = epoch_adversarial(test_loader, model_cnn_robust, pgd_linf)
    if t == 4:
        for param_group in opt.param_groups:
            param_group["lr"] = 1e-2
    print(*("{:.6f}".format(i) for i in (train_err, 1.0-adv_err, 100.0*adv_FPR, 100.0*adv_FNR, 1.0-test_err, 100.0* test_FPR, 100.0*test_FNR)), sep="\t")
torch.save(model_cnn_robust.state_dict(), "model_cnn_robust.pt")



print("FGSM: ", epoch_adversarial(test_loader, model_cnn_robust, fgsm))

print("PGD, 40 iter: ", epoch_adversarial(test_loader, model_cnn_robust, pgd_linf, num_iter=40))

print("pgd_linf_OT, 40 iter: ", epoch_adversarial_dro_eval(test_loader, model_cnn_robust, pgd_linf_OT_NoProjection2, 10.0, num_iter=40))


## Find wrongs
data_loader = DataLoader(mnist_test, batch_size = 10000, shuffle=False, num_workers=0)
for X,y in data_loader:
    X,y = X.to(device), y.to(device)
    break

def find_wrongs(X,y,yp):
    wrongs_idx=[]
    for i in range(10000):
            if yp[i].max(dim=0)[1] != y[i]:
                wrongs_idx.append(i)
    return wrongs_idx

### Illustrate attacked images on model robustified with FGSM
delta = pgd_linf(model_cnn_robust_fgsm, X, y, 0.1, num_iter=40)
yp = model_cnn_robust_fgsm(X + delta)

wrongs_idx_fgsm = find_wrongs(X, y, yp)

## Illustrate attacked images on model robustified with PGD
delta = pgd_linf(model_cnn_robust, X, y, 0.1, num_iter=40)
yp = model_cnn_robust(X + delta)

wrongs_idx_pgd = find_wrongs(X, y, yp)

print(wrongs_idx_fgsm)
print (len(wrongs_idx_fgsm))

print(wrongs_idx_pgd)
print (len(wrongs_idx_pgd))

wrongs_idx = wrongs_idx_fgsm + wrongs_idx_pgd
wrongs_list = [wrongs_idx_fgsm,wrongs_idx_pgd]

print('at least one of pgd or fgsm are wrong')
print(wrongs_idx)
print (len(wrongs_idx))
print (wrongs_list)

#########################################################
### Plotting for at least one wrong by either of the three (fgsm, pgd, and ours)
print ('at least one wrong by either of the three (fgsm, pgd, and ours ONLY FIRST MINIBATCH')
wrongs_idx=[259, 774, 8, 151, 412, 926, 936, 938, 813, 691, 829, 959, 65, 707, 965, 583, 844, 717, 591, 720, 726, 92, 866, 358, 495, 376, 381]

print ('at least one wrong by either of the three (fgsm, pgd, and ours ENTIRE DATASET')
wrongs_idx=[5634, 2052, 8, 1033, 1553, 18, 2578, 9745, 3601, 6166, 9749, 6168, 5654, 9755, 3100, 6173, 1062, 5159, 2043, 4140, 3117, 5676, 2607, 1072, 2098, 6706, 9280, 65, 583, 1609, 3146, 1611, 3662, 591, 2129, 1107, 9811, 1621, 2648, 7259, 92, 8287, 3681, 9316, 5734, 7783, 1128, 4201, 1641, 4207, 115, 9850, 8316, 5757, 2686, 9856, 2182, 646, 5769, 1678, 3726, 3727, 3730, 151, 4248, 2713, 1178, 4763, 1182, 2720, 2721, 7842, 9891, 9892, 7847, 684, 4783, 7856, 9904, 9905, 691, 1716, 1717, 2742, 6847, 707, 4294, 1737, 8397, 717, 4814, 720, 2769, 4306, 2771, 9422, 7886, 726, 3796, 2266, 1754, 7899, 2272, 4321, 4838, 3818, 7915, 2292, 1270, 2298, 2810, 2308, 8453, 774, 4360, 3336, 4874, 7434, 3850, 3853, 4880, 4380, 9500, 4382, 9505, 3373, 5936, 3893, 7990, 7991, 3384, 9019, 829, 2877, 3902, 9540, 4425, 2380, 844, 3405, 1871, 1364, 3926, 7514, 4443, 4956, 5985, 1378, 358, 359, 2406, 6505, 9071, 4978, 9587, 2422, 376, 7545, 2426, 2938, 3451, 381, 4477, 3457, 6532, 8069, 2952, 3976, 4497, 1425, 3474, 9620, 1429, 2454, 7574, 4504, 4505, 6035, 6042, 412, 3995, 2462, 6559, 6560, 8607, 926, 9634, 1955, 9638, 8102, 936, 938, 7595, 8110, 6065, 6578, 7094, 4536, 3005, 1981, 959, 1982, 8128, 6598, 6599, 5068, 2004, 1500, 5086, 4575, 5600, 4065, 9698, 9700, 2534, 3559, 6632, 2025, 1003, 4078, 495, 3567, 6641, 3062, 9211]


print(wrongs_idx)

random.seed(3)
rand_idx = random.sample(wrongs_idx, 12)
rand_idx = torch.tensor(rand_idx).to(device)
## rand_idx holds the ones that at least one of pgd or fgsm are wrong 
print ("random indices: ", rand_idx)

X_wrongs = torch.index_select(X, 0, rand_idx)
y_wrongs = torch.index_select(y, 0, rand_idx)

print("wrong ys: ", y_wrongs )

### Plotting
from matplotlib import pyplot as plt
import numpy as np

def plot_images(X,y,yp,M,N):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N,M*1.3))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(1-X[i*N+j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Pred: {}".format(yp[i*N+j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    plt.tight_layout()

### Illustrate original predictions
yp = model_cnn(X_wrongs)
plot_images(X_wrongs, y_wrongs, yp, 2, 6)

### Illustrate attacked images on plain model
delta = pgd_linf(model_cnn, X_wrongs, y_wrongs, 0.1, num_iter=40)
yp = model_cnn(X_wrongs + delta)
plot_images(X_wrongs+delta, y_wrongs, yp, 2, 6)


### Illustrate attacked images on model robustified with FGSM
delta = pgd_linf(model_cnn_robust_fgsm, X_wrongs, y_wrongs, 0.1, num_iter=40)
yp = model_cnn_robust_fgsm(X_wrongs + delta)
plot_images(X_wrongs+delta, y_wrongs, yp, 2, 6)

### Illustrate attacked images on model robustified with PGD
delta = pgd_linf(model_cnn_robust, X_wrongs, y_wrongs, 0.1, num_iter=40)
yp = model_cnn_robust(X_wrongs + delta)
plot_images(X_wrongs+delta, y_wrongs, yp, 2, 6)


print ("Done!")


