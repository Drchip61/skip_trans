# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:49:03 2021

@author: 严天宇
"""


import torch
import torch.nn as nn
import cv2

class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size,hidden_size)
        self.fc4 = nn.Linear(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x  = x.view(x.size(0),-1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
model = FCNet(784,500,10)


path = 'model.pth'

save_model = torch.load(path)
model_dict =  model.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)
model.load_state_dict(model_dict)


#model.load_state_dict(torch.load('model.pth'))


a = cv2.imread('photo.jpg',0)
a.resize(28,28)
a = torch.from_numpy(a).type(torch.FloatTensor)
a = a.unsqueeze(0)
b = model(a)
_, predict = torch.max(b.data, 1)
print(predict)
