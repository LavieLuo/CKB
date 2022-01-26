# -*- coding: utf-8 -*-

import network
import data_loader
import CKButils
import numpy as np
import scipy.io as io
from torch.utils import model_zoo
import torch
from datetime import datetime
import sys
import os
import argparse


##################################
# Set Parameter
##################################
# Your own path
data_folder = './dataset' # path of dataset
save_folder = './results' # path for saving results
log_folder = './logs' # path for saving logs
model_dir = './pre_model/' # path of pretrained model

##################################
# Experiment Main Function
##################################
def Experiment_Main(config):
    # Parameter
    dataset = config['dataset']
    net = config['net']
    model = config['model']
    FC_dim_1 = int(config['FC_dim_1'])
    FC_dim_2 = int(config['FC_dim_2'])
    exp_times = int(config['exp_times'])
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])
    Tar_Ent_lambda = float(config['Tar_Ent_lambda'])
    CKB_lambda = float(config['CKB_lambda'])
    CKB_type = config['CKB_type']
    Tar_Ent_epoch = int(config['Tar_Ent_epoch'])
    CKB_epoch = int(config['CKB_epoch'])
    inv_epsilon = float(config['inv_epsilon'])
    lr = float(config['lr'])
    optim_param = config['optim_param']
    
    if os.path.exists(log_folder) is False:
        os.mkdir(log_folder)
    log_name = '%1s_%1s_%1s_EntEpoch%1s_EntLam%1s_CKBEpoch%1s_CKBLam%1s_epsilon%1s_lr%1s.txt'%(dataset,net,model,Tar_Ent_epoch,Tar_Ent_lambda,CKB_epoch,CKB_lambda,inv_epsilon,lr)
    log_file = open(os.path.join(log_folder,log_name), "w")
    
    ##################################
    # Prepare Data
    ##################################
    if net == 'AlexNet':
        alexnet = True
    else:
        alexnet = False
    source_domain_set, target_domain_set, task_set, num_cls = data_loader.get_tasks(dataset)
    Source_Acc_Recorder = np.zeros((exp_times,len(task_set)))
    Target_Acc_Recorder = np.zeros((exp_times,len(task_set)))
    for task_iter in range(len(task_set)):
        source_domain = source_domain_set[task_iter]
        target_domain = target_domain_set[task_iter]
        task = task_set[task_iter]
        # Training loader
        source_tr_loader = data_loader.loader(dataset,data_folder,source_domain,
                                              batch_size,alexnet,train=True)
        target_tr_loader = data_loader.loader(dataset,data_folder,target_domain,
                                              batch_size,alexnet,train=True)
        # Testing loader
        source_te_loader = data_loader.loader(dataset,data_folder,source_domain,
                                              batch_size,alexnet,train=False)
        target_te_loader = data_loader.loader(dataset,data_folder,target_domain,
                                              batch_size,alexnet,train=False)
        ##################################
        # Random Experiments
        ##################################
        for exp_iter in range(exp_times):
            ##################################
            # Initialize network and optimizer
            ##################################
            # Network
            if net == 'AlexNet':
                # ImageNet pretrained AlexNet
                DNN = network.AlexNet_Feature()
                FC_input_dim = 4096
            elif net == 'ResNet-50':
                # ImageNet pretrained ResNet
                if os.path.exists(model_dir) is False:
                    os.mkdir(model_dir)
                DNN = network.ResNet50(network.Bottleneck, [3, 4, 6, 3])
                FC_input_dim = 2048
                url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
                pretrained_dict = model_zoo.load_url(url,model_dir)
                del pretrained_dict['fc.bias']
                del pretrained_dict['fc.weight']
                DNN.load_state_dict(pretrained_dict)
                del pretrained_dict
            else:
                sys.exit('Error: invalid network')
                
            FC = network.FC_Layers(FC_input_dim,FC_dim_1,FC_dim_2,num_cls)
            FC.apply(CKButils.weights_init)
            FC.cuda()
            DNN.cuda()
            
            # Optimizer
            if optim_param == 'GD':
                optimizer_dict = [{"params": filter(lambda p: p.requires_grad, DNN.parameters()), "lr": 1},
                          {"params": filter(lambda p: p.requires_grad, FC.parameters()), "lr": 10}]
                optimizer = torch.optim.SGD(optimizer_dict, lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
                param_lr = []
                for param_group in optimizer.param_groups:
                    param_lr.append(param_group["lr"])
            elif optim_param == 'Adam':
                beta1=0.9
                beta2=0.999
                optimizer = torch.optim.Adam([{'params':DNN.parameters(), 'lr': lr*0.1},
                                        {'params':FC.parameters()}], lr*1.5,
                                        [beta1, beta2], weight_decay=0.01)
            else:
                sys.exit('Error: invalid optimizer')
            
            ####################
            # Training 
            ####################
            iter_optim = 1
            for step in range(epochs):
                epoch_time_start = datetime.now()
                ####################
                # Mini-batch training 
                ####################
                for (X_s, lab_s), (X_t, lab_t) in zip(source_tr_loader,target_tr_loader):
                    # switch to training mode
                    DNN.train()
                    FC.train()
                    
                    if optim_param == 'GD':
                    # upadate lr
                        optimizer = CKButils.inv_lr_scheduler(param_lr, optimizer, iter_optim, init_lr=lr, gamma=0.001, power=0.75,
                                          weight_decay=0.0005)
                        iter_optim += 1
                    
                    # load data
                    X_s, lab_s = CKButils.to_var(X_s), CKButils.to_var(lab_s)
                    X_t, lab_t = CKButils.to_var(X_t), CKButils.to_var(lab_t)
                    
                    # Init gradients
                    DNN.zero_grad()
                    FC.zero_grad()
                    
                    # Forward propagate
                    Z_s, prob_s = FC(DNN(X_s))
                    Z_t, prob_t = FC(DNN(X_t))
                    plab_t = prob_t.detach().max(1)[1]
                    
                    # norm_s = Z_s.pow(2).detach().sum(1).pow(1/2).unsqueeze(1)
                    # norm_t = Z_t.pow(2).detach().sum(1).pow(1/2).unsqueeze(1)    
                    # Z_s = (Z_s.mul(1/norm_s))*1e1
                    # Z_t = (Z_t.mul(1/norm_t))*1e1
                    
                    ####################
                    # Loss Objective
                    ####################
                    # Cross-Entropy
                    CE_loss = CKButils.Cross_Entropy(prob_s, lab_s)
                    
                    # Entropy
                    if step <= (Tar_Ent_epoch - 1):
                        Tar_Ent_loss = torch.zeros(1).squeeze(0).cuda()
                    else:
                        Tar_Ent_loss = Tar_Ent_lambda*CKButils.Entropy(prob_t)
                    
                    # CKB Matching Loss
                    if step <= (CKB_epoch - 1):
                        Match_loss = torch.zeros(1).squeeze(0).cuda()
                    else:
                        if model == 'CKB':
                            CKB_loss = CKB_lambda*CKButils.CKB_Metric(Z_s,Z_t,lab_s,plab_t,prob_t,num_cls,inv_epsilon,CKB_type)
                            Match_loss = CKB_loss
                        elif model == 'CKB+MMD':
                            OneHot_s = torch.zeros(lab_s.shape[0],num_cls).cuda().scatter(1,lab_s.unsqueeze(1),1).detach()
                            CKB_loss = CKB_lambda*CKButils.CKB_Metric(Z_s,Z_t,lab_s,plab_t,prob_t,num_cls,inv_epsilon,CKB_type)
                            MMD_y_loss = CKB_lambda*CKButils.MMD_Metric(OneHot_s,prob_t)
                            Match_loss = CKB_loss + MMD_y_loss
                        else:
                            sys.exit('Error: invalid model')
                    
                    # Overall Loss
                    O_loss = CE_loss + Tar_Ent_loss + Match_loss
                    
                    # Backward propagate
                    O_loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    
                ####################
                # Testing 
                ####################
                # switch to testing mode
                DNN.eval()
                FC.eval()
                # evaluate Model
                source_acc = CKButils.classification_accuracy(source_te_loader,DNN,FC)*100
                target_acc = CKButils.classification_accuracy(target_te_loader,DNN,FC)*100
                
                ####################
                # Report results
                ####################
                # time
                epoch_time_end = datetime.now()
                seconds = (epoch_time_end - epoch_time_start).seconds
                minutes = seconds//60
                second = seconds%60
                hours = minutes//60
                minute = minutes%60
                # print result
                print('====================== [%1s] %1s→%1s: Experiment %1s Epoch %1s ================='%(dataset,source_domain,target_domain,exp_iter+1,step+1))
                print('Source Accuracy: %1s'%source_acc)
                print('Target Accuracy: %1s'%target_acc)
                print('Cross-Entropy Loss: %1s'%(CE_loss.data.data.cpu().numpy()))
                print('Target Entropy Loss: %1s'%(Tar_Ent_loss.data.cpu().numpy()))
                print('Matching Loss: %1s'%(Match_loss.data.cpu().numpy()))
                print('Overall Loss: %1s'%(O_loss.data.cpu().numpy()))
                print('Current epoch [train & test] time cost: %1s Hour %1s Minutes %1s Seconds'%(hours,minute,second))
                if target_acc == 1:
                    print('Reach accuracy {1} at Epoch %1s !'%(step+1))
                    break
                # write log file
                log_str = '%1s | [%1s] %1s→%1s: Experiment %1s Epoch %1s, target accuracy %1s:'%(epoch_time_end,dataset,source_domain,target_domain,exp_iter+1,step+1,target_acc)
                log_file.write(log_str+'\n')
                log_file.flush()
                # empty network cache
                torch.cuda.empty_cache()
                
            ####################
            # Record results
            ####################
            Source_Acc_Recorder[exp_iter,task_iter] = source_acc
            Target_Acc_Recorder[exp_iter,task_iter] = target_acc
            
    ####################
    # Save results
    ####################
    if os.path.exists(save_folder) is False:
        os.mkdir(save_folder)
    result_dict = config
    result_dict['Source_ACC'] = Source_Acc_Recorder
    result_dict['Target_ACC'] = Target_Acc_Recorder
    file_name = '%1s_%1s_%1s_EntEpoch%1s_EntLam%1s_CKBEpoch%1s_CKBLam%1s_epsilon%1s_lr%1s.mat'%(dataset,net,model,Tar_Ent_epoch,Tar_Ent_lambda,CKB_epoch,CKB_lambda,inv_epsilon,lr)
    
    io.savemat(os.path.join(save_folder,file_name),result_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Conditional Kernel Bures Metric CVPR-2021')
    parser.add_argument('--dataset', type=str, default='ImageCLEF', choices=['ImageCLEF', 'OfficeHome', 'Office10', 'RefurbishedOffice31'])
    parser.add_argument('--net', type=str, default='ResNet-50', choices=['ResNet-50', 'AlexNet'])
    parser.add_argument('--model', type=str, default='CKB', choices=['CKB', 'CKB+MMD'])
    parser.add_argument('--FC_dim_1', type=str, default='1024', help="dimension of the 1st FC layer")
    parser.add_argument('--FC_dim_2', type=str, default='512', help="dimension of the 2nd FC layer")
    parser.add_argument('--exp_times', type=str, default='10', help="numbers of random experiment")
    parser.add_argument('--epochs', type=str, default='150', help="maximum training epochs")
    parser.add_argument('--batch_size', type=str, default='40', help="training batch_size; 40 for ResNet-50 and 128 for AlexNet")
    parser.add_argument('--Tar_Ent_lambda', type=str, default='5e-2', help="lambda_1 in paper")
    parser.add_argument('--CKB_lambda', type=str, default='1e0', help="lambda_2 in paper")
    parser.add_argument('--CKB_type', type=str, default='soft', help="target soft/hard labels")
    parser.add_argument('--Tar_Ent_epoch', type=str, default='10', help="training with target entropy loss after # epochs")
    parser.add_argument('--CKB_epoch', type=str, default='5', help="training with CKB loss after # epochs")
    parser.add_argument('--inv_epsilon', type=str, default='1e-2', help="regularization parameter of kernel matrix inverse")
    parser.add_argument('--lr', type=str, default='2e-4', help="learning rate")
    parser.add_argument('--optim_param', type=str, default='Adam', choices=['Adam', 'GD'])
    parser.add_argument('--GPU_device', type=str, nargs='?', default='0', help="set GPU device for training")
    parser.add_argument('--seed', type=str, default='0', help="random seed")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_device
    
    config = {}
    config['dataset'] = args.dataset
    config['net'] = args.net
    config['model'] = args.model
    config['FC_dim_1'] = args.FC_dim_1
    config['FC_dim_2'] = args.FC_dim_2
    config['exp_times'] = args.exp_times
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['Tar_Ent_lambda'] = args.Tar_Ent_lambda
    config['CKB_lambda'] = args.CKB_lambda
    config['CKB_type'] = args.CKB_type
    config['Tar_Ent_epoch'] = args.Tar_Ent_epoch
    config['CKB_epoch'] = args.CKB_epoch
    config['inv_epsilon'] = args.inv_epsilon
    config['lr'] = args.lr
    config['optim_param'] = args.optim_param
    config['GPU_device'] = args.GPU_device
        
    ##################################
    # Random Seeds
    ##################################
    torch.manual_seed(int(args.seed)) 
    # Run experiments
    Experiment_Main(config)
