#!/usr/bin/env python
# coding=utf-8
import os
import random
import argparse

import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from PIL import Image

from net import SSD300
from loss import SSDLoss


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default='./examples/ssd/model/ssd512_vgg16.pth', type=str, help='initialized model path')
parser.add_argument('--checkpoint', default='./examples/ssd/checkpoint/ckpt.pth', type=str, help='checkpoint path')
args = parser.parse_args()

print('==> Building model..')
net = SSD300(num_classes=21)
net.load_state_dict(torch.load(args.model))
best_loss = float('inf')
start_epoch = 0 
if arg.resume:
    


def train(epoch):
    print('epoch: %d\n' % epoch)
    net.train()
    train_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        # inputs = Variable(inputs.cuda())
        # loc_targets = Variable(loc_targets.cuda())
        # cls_targets = Variable(cls_targets.cuda())
        inputs = inputs.cuda()
        loc_targets = loc_targets.cuda()
        cls_targets = cls_targets.cuda()

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backwords()
        optimizer.step()

        train_loss += loss.data[0]
        print('train_loss: %.3f | avg_loss: %.3f [%d/%d]'  % (loss.data[0], train_loss/(batch_idx+1), batch_idx+1, len(trainloader)))



def test():



if __name__ == '__main__':
   


