"""
@Time    : 2021/10/22 10:30
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@Software: PyCharm
LOAD DATA from file.
"""

##
import os
import torch
import numpy as np
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from load_data.preprocessing import main_process_unsw
from load_data.process_nslkdd import main_process_nsl
from load_data.process_cicids import main_process_cicids
from cicdos2019 import process_dos
from torch.utils.data import DataLoader,TensorDataset
import torchvision.transforms as transforms

##
def load_data(opt):
    """ Load Data

    Args:
        opt ([type]): Argument Parser

    Raises:
        IOError: Cannot Load Dataset

    Returns:
        [type]: dataloader
    """

    ##
    # LOAD DATA SET
    if opt.dataroot == '':
        opt.dataroot = './data/{}'.format(opt.dataset)

    if opt.dataset in ['unsw']:
        data, x_test, y_test = main_process_unsw(100)
        y_train = np.zeros(len(data))
        y_train_tensor = torch.FloatTensor(y_train)
        x_test_tensor = torch.FloatTensor(x_test)
        X_normal = torch.FloatTensor(data)
        y_test_tensor = torch.FloatTensor(y_test)
        X_normal = TensorDataset(X_normal,y_train_tensor)
        test_set = TensorDataset(x_test_tensor,y_test_tensor)
        train_loader = torch.utils.data.DataLoader(dataset=X_normal,
                                  batch_size=64,
                                  shuffle=True,
                                drop_last=True )
        testload = torch.utils.data.DataLoader(dataset=test_set,
                                  batch_size=64,
                                  shuffle=True,
                                drop_last=True )
        return train_loader,testload
    elif opt.dataset in ['nsl']:
        data, x_test, y_test = main_process_nsl(100)
        y_train = np.zeros(len(data))
        y_train_tensor = torch.FloatTensor(y_train)

        x_test_tensor = torch.FloatTensor(x_test)
        X_normal = torch.FloatTensor(data)
        y_test_tensor = torch.FloatTensor(y_test)
        X_normal = TensorDataset(X_normal, y_train_tensor)  # 对tensor进行打包
        test_set = TensorDataset(x_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(dataset=X_normal,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   drop_last=True)
        testload = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=64,
                                               shuffle=True,
                                               drop_last=True)
        return train_loader,testload

    elif opt.dataset in ['cicids']:
        data, x_test, y_test,len_normal,len_malware = main_process_cicids(100)
        y_train = np.zeros(len(data))
        y_train_tensor = torch.FloatTensor(y_train)

        x_test_tensor = torch.FloatTensor(x_test)
        X_normal = torch.FloatTensor(data)
        y_test_tensor = torch.FloatTensor(y_test)
        X_normal = TensorDataset(X_normal, y_train_tensor)  # 对tensor进行打包
        test_set = TensorDataset(x_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(dataset=X_normal,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   drop_last=True)
        testload = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=64,
                                               shuffle=True,
                                               drop_last=True)
        return train_loader, testload
    elif opt.dataset in ['dos']:
        train_normal,x_test,y_test = process_dos()
        y_train = np.zeros(len(train_normal))
        y_train_tensor = torch.FloatTensor(y_train)
        #y_test = np.ones(len(x_test))
        y_test_tensor = torch.FloatTensor(y_test)

        x_test_tensor = torch.FloatTensor(x_test)

        X_normal = torch.FloatTensor(train_normal)

        X_normal = TensorDataset(X_normal, y_train_tensor)  # 对tensor进行打包
        test_set = TensorDataset(x_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(dataset=X_normal,
                                                   batch_size=64,
                                                   shuffle=True,
                                                   drop_last=True)
        testload = torch.utils.data.DataLoader(dataset=test_set,
                                               batch_size=64,
                                               shuffle=True,
                                               drop_last=True)
        return train_loader,testload

    else:
        splits = ['train', 'test']
        drop_last_batch = {'train': True, 'test': False}
        shuffle = {'train': True, 'test': True}
        transform = transforms.Compose([transforms.Resize(opt.isize),
                                        transforms.CenterCrop(opt.isize),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), transform) for x in splits}
        dataloader = {x: torch.utils.data.DataLoader(dataset=dataset[x],
                                                     batch_size=opt.batchsize,
                                                     shuffle=shuffle[x],
                                                     num_workers=int(opt.workers),
                                                     drop_last=drop_last_batch[x],
                                                     worker_init_fn=(None if opt.manualseed == -1
                                                     else lambda x: np.random.seed(opt.manualseed)))
                      for x in splits}
        return dataloader