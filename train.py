"""
@Time    : 2021/10/22 10:30
-------------------------------------------------
@Author  : sailorlee(lizeyi)
@email   : chrislistudy@163.com
-------------------------------------------------
@FileName: train.py
@Software: PyCharm

TRAIN FlowGANAnomaly
. Example: Run the following command from the terminal.
    run train.py                             \
        --model FlowGANAnomaly                        \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function
from options import Options
from lib.data import load_data
from lib.model import FlowGANAnomaly

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
    model = FlowGANAnomaly(opt, train_loader)
    ##
    # TEST : get roc auc
    # model.test_malware(testload)


    ##
    # TRAIN MODEL
    model.train(testload)

if __name__ == '__main__':

    train()
