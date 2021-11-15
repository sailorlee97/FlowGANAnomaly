"""
@Time    : 2021/10/22 10:30
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: plt_box_total.py
@Software: PyCharm

TRAIN GANOMALY
. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function
from torch.utils.data import DataLoader,TensorDataset
from options import Options
from lib.data import load_data
from lib.model import Ganomaly
from load_data.process_nslkdd import main_process_nsl
from eva.plot_culve import plot_tsne
import torch
from load_data.preprocessing import main_process_unsw
##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()
    ##
    # LOAD DATA
    train_loader, testload= load_data(opt)

    ##
    # LOAD MODEL
    model = Ganomaly(opt, train_loader)
    ##
    # TEST : get roc auc
    # model.test_malware(testload)


    ##
    # TRAIN MODEL
    model.train(testload)

if __name__ == '__main__':

    train()
