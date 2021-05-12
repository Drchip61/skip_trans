# -*- coding: utf-8 -*-
"""
Created on Tue May 11 14:29:29 2021

@author: 严天宇
"""
import torch
import torch.nn as nn
from DUTS_dataset import MyData
from net_try import DiceLoss,try_net
from torch.utils.data import DataLoader
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = try_net().to(device)
ce_loss = nn.CrossEntropyLoss()
dice_loss = DiceLoss(2)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)


train_loader = DataLoader(MyData(),
                      shuffle=True,
                      batch_size=50)

a = []

for epoch_num in range(1):
    for i_batch,(img,label) in enumerate(train_loader):
        image_batch, label_batch = img.to(device), label.to(device)
        outputs = model(image_batch)
        loss_ce = ce_loss(outputs, label_batch.long())
        #loss_dice = dice_loss(outputs, label_batch, softmax=True)
        loss =  loss_ce
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
        
        if i_batch == 3:
            break
    
    a.append(outputs.tolist())
    
