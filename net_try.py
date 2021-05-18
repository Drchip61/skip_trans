# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:17:47 2021

@author: 严天宇
"""

import torch
import torch.nn as nn
import numpy as np
import torch

import torch.nn as nn

class try_net(nn.Module):
    def __init__(self):
        super(try_net,self).__init__()
        self.conv1_1 = nn.Conv2d(3,64,3,1,1)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.conv1_3 = nn.Conv2d(128,64,3,1,1)
        
        self.conv2_1 = nn.Conv2d(64,128,3,1,1)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.conv2_3 = nn.Conv2d(256,128,3,1,1)
        
        self.conv3_1 = nn.Conv2d(128,256,3,1,1)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.conv3_3 = nn.Conv2d(512,256,3,1,1)
        
        self.conv4_1 = nn.Conv2d(256,512,3,1,1)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.conv4_3 = nn.Conv2d(1024,512,3,1,1)
        
        self.pool = nn.MaxPool2d(2)
        self.deconv1 = nn.ConvTranspose2d(512,512,2,2,0)
        self.deconv2 = nn.ConvTranspose2d(512,256,2,2,0)
        self.deconv3 = nn.ConvTranspose2d(256,128,2,2,0)
        self.deconv4 = nn.ConvTranspose2d(128,64,2,2,0)
        
        self.conv_more = nn.Conv2d(64,2,3,1,1)
        
        self.relu = nn.ReLU()
        
        self.bn_1 = nn.BatchNorm2d(128)
        self.bn_2 = nn.BatchNorm2d(256)
        self.bn_3 = nn.BatchNorm2d(512)
        self.bn_4 = nn.BatchNorm2d(1024)
        
    
   
        
    def forward(self,x):
        local = []
        
        
        x = self.conv1_1(x)
        x = self.relu(x)
        x = self.conv1_2(x)#64
        x = self.relu(x)
        local.append(x.contiguous())
        x = self.pool(x)#64*128*128
        
        
        x = self.conv2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        local.append(x.contiguous())
        x = self.pool(x)#256*256*64
        x = self.bn_1(x)
        
        
        x = self.conv3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        local.append(x.contiguous())
        x = self.pool(x)#32
        x = self.bn_2(x)
        
        x = self.conv4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.relu(x)
        local.append(x.contiguous())
        x = self.pool(x)#16
        x = self.bn_3(x)
        
        x = self.deconv1(x)#256
        x = self.relu(x)
        x = torch.cat([x,local[3]],dim = 1)
        x = self.bn_4(x)
        x = self.conv4_3(x)
        x = self.relu(x)
        x = self.conv4_2(x)#512
        x = self.relu(x)
        
        
        x = self.deconv2(x)#128
        x = self.relu(x)
        x = torch.cat([x,local[2]],dim = 1)#512
        x = self.conv3_3(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.relu(x)
        #256
        
        
        x = self.deconv3(x)
        x = self.relu(x)
        x = torch.cat([x,local[1]],dim = 1)#256
        x = self.conv2_3(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.relu(x)
        
        x = self.deconv4(x)
        x = self.relu(x)
        x = torch.cat([x,local[0]],dim = 1)#128
        x = self.conv1_3(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        
        x = self.conv_more(x)
        
        return x
    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes