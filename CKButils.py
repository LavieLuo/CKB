# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import sys

##################################
# Network & Variable
##################################
def weights_init(m):
    """Initialize network parameters."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)         
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    
def to_var(x):
    """Convert numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()   

def classification_accuracy(data_loader,DNN,FC):
    with torch.no_grad():
        correct = 0
        for batch_idx, (X, lab) in enumerate(data_loader):
            X, lab = to_var(X), to_var(lab).long().squeeze()
            _, prob = FC(DNN(X))
            plab = prob.data.max(1)[1]
            correct += plab.eq(lab.data).cpu().sum()
        accuracy = correct.item() / len(data_loader.dataset)
        return accuracy
    
    
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


schedule_dict = {"inv":inv_lr_scheduler}

##################################
# Objective Functions
##################################
# Cross-Entropy Loss
NLL_loss = torch.nn.NLLLoss().cuda() 
def Cross_Entropy(prob,lab):
    CE_loss = NLL_loss(torch.log(prob+1e-4), lab)
    return CE_loss

# Entropy Loss
def Entropy(prob):
    num_sam = prob.shape[0]
    Entropy = -(prob.mul(prob.log()+1e-4)).sum()
    return Entropy/num_sam

# CKB loss
def CKB_Metric(fea_s, fea_t, lab_s, plab_t, prob_t, num_cls, epsilon=1e-2, CKB_type='soft'):
#   Y: label, Z: fea, matching conditional distribution P(Z|Y)
    num_sam_s = fea_s.shape[0]
    num_sam_t = fea_t.shape[0]
    OneHot_s = torch.zeros(num_sam_s,num_cls).cuda().scatter(1,lab_s.unsqueeze(1),1).detach()
    if CKB_type == 'hard':
        prob_t = torch.zeros(num_sam_t,num_cls).cuda().scatter(1,plab_t.unsqueeze(1),1).detach()
    elif CKB_type == 'soft':
        prob_t = prob_t.detach()
    else:
        sys.exit('Error: invalid CKB_type')
    I_s = torch.eye(num_sam_s).cuda()
    I_t = torch.eye(num_sam_t).cuda()
    
    #====== Kernel Matrix and Centering Matrix =======
    H_s = ( torch.eye(num_sam_s) - torch.ones(num_sam_s)/num_sam_s ).cuda()
    H_t = ( torch.eye(num_sam_t) - torch.ones(num_sam_t)/num_sam_t ).cuda()
    
    D_YsYs = OneHot_s.pow(2).sum(1,keepdim=True).repeat(1,num_sam_s) +\
           OneHot_s.pow(2).sum(1,keepdim=True).t().repeat(num_sam_s,1) -\
           2*torch.mm(OneHot_s,OneHot_s.t())
    D_YtYt = prob_t.pow(2).sum(1,keepdim=True).repeat(1,num_sam_t) +\
           prob_t.pow(2).sum(1,keepdim=True).t().repeat(num_sam_t,1) -\
           2*torch.mm(prob_t,prob_t.t())
    D_ZsZs = fea_s.pow(2).sum(1,keepdim=True).repeat(1,num_sam_s) +\
           fea_s.pow(2).sum(1,keepdim=True).t().repeat(num_sam_s,1) -\
           2*torch.mm(fea_s,fea_s.t())
    D_ZtZt = fea_t.pow(2).sum(1,keepdim=True).repeat(1,num_sam_t) +\
           fea_t.pow(2).sum(1,keepdim=True).t().repeat(num_sam_t,1) -\
           2*torch.mm(fea_t,fea_t.t())
    D_ZtZs = fea_t.pow(2).sum(1,keepdim=True).repeat(1,num_sam_s) +\
           fea_s.pow(2).sum(1,keepdim=True).t().repeat(num_sam_t,1) -\
           2*torch.mm(fea_t,fea_s.t())
           
    sigma_YsYs = D_YsYs.mean().detach()
    sigma_YtYt = D_YtYt.mean().detach()
    sigma_ZsZs = D_ZsZs.mean().detach()
    sigma_ZtZt = D_ZtZt.mean().detach()
    sigma_ZtZs = D_ZtZs.mean().detach()
    
    K_YsYs = (-D_YsYs/sigma_YsYs).exp()
    K_YtYt = (-D_YtYt/sigma_YtYt).exp()
    K_ZsZs = (-D_ZsZs/sigma_ZsZs).exp()
    K_ZtZt = (-D_ZtZt/sigma_ZtZt).exp()
    K_ZtZs = (-D_ZtZs/sigma_ZtZs).exp()   
    
    G_Ys = (H_s.mm(K_YsYs)).mm(H_s)
    G_Yt = (H_t.mm(K_YtYt)).mm(H_t)
    G_Zs = (H_s.mm(K_ZsZs)).mm(H_s)
    G_Zt = (H_t.mm(K_ZtZt)).mm(H_t)
        
    #====== R_{s} and R_{t} =======
    Inv_s = (epsilon*num_sam_s*I_s + G_Ys).inverse()
    Inv_t = (epsilon*num_sam_t*I_t + G_Yt).inverse()
    R_s = epsilon*G_Zs.mm(Inv_s)
    R_t = epsilon*G_Zt.mm(Inv_t)
       
    #====== R_{st} =======
    # B_s = I_s - (G_Ys - (G_Ys.mm(Inv_s)).mm(G_Ys))/(num_sam_s*epsilon)
    # B_t = I_t - (G_Yt - (G_Yt.mm(Inv_t)).mm(G_Yt))/(num_sam_t*epsilon)
    B_s = num_sam_s*epsilon*Inv_s
    B_t = num_sam_t*epsilon*Inv_t
    B_s = (B_s + B_s.t())/2 # numerical symmetrize
    B_t = (B_t + B_t.t())/2 # numerical symmetrize
    S_s, U_s = B_s.symeig(eigenvectors=True)
    S_t, U_t = B_t.symeig(eigenvectors=True)
    HC_s = H_s.mm( U_s.mm((S_s+1e-4).pow(0.5).diag()) )
    HC_t = H_t.mm( U_t.mm((S_t+1e-4).pow(0.5).diag()) )
    Nuclear = (HC_t.t().mm(K_ZtZs)).mm(HC_s)
    U_n, S_n, V_n = torch.svd(Nuclear)
    
    #====== Conditional KB Distance
    CKB_dist = R_s.trace() + R_t.trace() - 2*S_n[:-1].sum()/((num_sam_s*num_sam_t)**0.5)
    
    return CKB_dist

# MMD loss
def MMD_Metric(prob_s, prob_t):
    num_sam_s = prob_s.shape[0]
    num_sam_t = prob_t.shape[0]
    D_XsXs = prob_s.pow(2).sum(1,keepdim=True).repeat(1,num_sam_s) +\
           prob_s.pow(2).sum(1,keepdim=True).t().repeat(num_sam_s,1) -\
           2*torch.mm(prob_s,prob_s.t())
    D_XtXt = prob_t.pow(2).sum(1,keepdim=True).repeat(1,num_sam_t) +\
           prob_t.pow(2).sum(1,keepdim=True).t().repeat(num_sam_t,1) -\
           2*torch.mm(prob_t,prob_t.t())
    D_XtXs = prob_t.pow(2).sum(1,keepdim=True).repeat(1,num_sam_s) +\
           prob_s.pow(2).sum(1,keepdim=True).t().repeat(num_sam_t,1) -\
           2*torch.mm(prob_t,prob_s.t())
    
    sigma_XsXs = D_XsXs.mean().detach()
    sigma_XtXt = D_XtXt.mean().detach()
    sigma_XtXs = D_XtXs.mean().detach()
        
    K_XsXs = (-D_XsXs/sigma_XsXs).exp()
    K_XtXt = (-D_XtXt/sigma_XtXt).exp()
    K_XtXs = (-D_XtXs/sigma_XtXs).exp()
    
    MMD_dist = K_XsXs.mean() + K_XtXt.mean() - 2*K_XtXs.mean()
    return MMD_dist
