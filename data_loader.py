# -*- coding: utf-8 -*-

import sys
import os
from torchvision import datasets, transforms
import torch.utils.data as data


##################################
# Load transfer tasks
##################################
def get_tasks(dset):
    if dset == 'ImageCLEF':
        source_domain_set = ['I','P','I','C','C','P']
        target_domain_set = ['P','I','C','I','P','C']
        task_set = ['ItoP','PtoI','ItoC','CtoI','CtoP','PtoC']
        num_cls = 12
    elif dset == 'OfficeHome':
        source_domain_set = ['Art','Art','Art','Clipart','Clipart','Clipart','Product','Product','Product','Real_World','Real_World','Real_World']
        target_domain_set = ['Clipart','Product','Real_World','Art','Product','Real_World','Art','Clipart','Real_World','Art','Clipart','Product']
        task_set = ['ARtoCL','ARtoPR','ARtoRW','CLtoAR','CLtoPR','CLtoRW','PRtoAR','PRtoCL','PRtoRW','RWtoAR','RWtoCL','RWtoPR']
        num_cls = 65
    elif dset == 'Office10':
        source_domain_set = ['A','A','A','C','C','C','D','D','D','W','W','W']
        target_domain_set = ['C','D','W','A','D','W','A','C','W','A','C','D']
        task_set = ['AtoW','AtoD','AtoC','WtoA','WtoD','WtoC','DtoA','DtoW','DtoC','CtoA','CtoW','CtoD']
        num_cls = 10
    elif dset == 'RefurbishedOffice31':
        source_domain_set = ['A','A','W','W','D','D']
        target_domain_set = ['W','D','A','D','A','W']
        task_set = ['AtoW','AtoD','WtoA','WtoD','DtoA','DtoW']
        num_cls = 31
    else:
        sys.exit('Error: invalid dataset name')
        
    return source_domain_set, target_domain_set, task_set, num_cls


##################################
# Get folder path of domain
##################################
def get_path(dset,data_folder,domain):
    if dset == 'ImageCLEF':
        dset = 'Image_CLEF'
        domain = domain.lower()
    elif dset == 'OfficeHome':
        domain = domain
    elif (dset == 'Office10')|(dset == 'RefurbishedOffice31'):
        if domain == 'A':
            domain = 'amazon'
        elif domain == 'C':
            domain = 'caltech'
        elif domain == 'D':
            domain = 'dslr'
        elif domain == 'W':
            domain = 'webcam'
    if dset == 'RefurbishedOffice31':
        dset = 'Modern-Office-31'
        
    if dset == 'Office10':
        data_path = os.path.join(data_folder,dset,domain,'images')
    else:
        data_path = os.path.join(data_folder,dset,domain)
        
    return data_path
    


##################################
# Dataloader
##################################
# simple loader
def loader(dset,data_folder,domain,batch_size,alexnet=False,train=True):
    data_path = get_path(dset,data_folder,domain)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    if alexnet:
        size_1 = 256
        size_crop = 227
    else:
        size_1 = 256
        size_crop = 224
    if train:
        transform = [
            transforms.Resize((size_1, size_1)),
            transforms.Resize((size_crop, size_crop)),
            # transforms.RandomResizedCrop(size_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
        shuffle = True
    else:
        transform = [
            transforms.Resize((size_1, size_1)),
            transforms.Resize((size_crop, size_crop)),
            # transforms.CenterCrop((size_crop,size_crop)),
            transforms.ToTensor(),
            normalize,
        ]
        shuffle = False
    data_loader = data.DataLoader(
        dataset=datasets.ImageFolder(
            data_path,
            transform=transforms.Compose(transform)
        ),
        batch_size=batch_size,num_workers=4,
        shuffle=shuffle,
    )
    
    return data_loader