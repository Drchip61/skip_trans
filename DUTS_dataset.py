# -*- coding: utf-8 -*-
"""
Created on Tue May 11 00:27:46 2021

@author: 严天宇
"""


import scipy.io as io
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import pdb
import os
from PIL import Image
import numpy as np
import os
import cv2



class MyData(Dataset):
    def __init__(self):
        super(MyData, self).__init__()
        self.train_im_path = 'E:\PYTHON\DUTS-TR\DUTS-TR\DUTS-TR-Image'
        self.train_lb_path = 'E:\PYTHON\DUTS-TR\DUTS-TR\DUTS-TR-Mask'
        self.train_im_num = 10553
        self.train_imgs = os.listdir('E:\PYTHON\DUTS-TR\DUTS-TR\DUTS-TR-Image')
        self.train_labels = os.listdir('E:\PYTHON\DUTS-TR\DUTS-TR\DUTS-TR-Mask')

    def __len__(self):
        return self.train_im_num

    def __getitem__(self, index):#index是里面最具有特色的参数，默认遍历整个len的长度！！
        #所以这里可以视为拿到了所有被处理的数据集，而且巧妙的是，还被封装成dataset 的形式！！！！！
        ##对每一个img和label进行处理，从而得到的是每一个被处理后的item
        # load image
        img_file = os.path.join(self.train_im_path,self.train_imgs[index])
        img = Image.open(img_file)
        label_file = os.path.join(self.train_lb_path,self.train_labels[index])
        labels = Image.open(label_file)
        #labels = np.array(labels.getdata()).reshape(1,labels.size[0], labels.size[1])
        labels = self.transform_2(labels)
        
        for i in range(labels.size(0)):
            for j in range(labels.size(1)):
                for k in range(labels.size(2)):
                    if labels[i][j][k] >= 0.1:
                        labels[i][j][k] = 1.0
                    else:
                        labels[i][j][k] = 0.0
                    
        im = self.transform_1(img)
        
        lb = self.transform_3(labels)
        '''
        for i in range(lb.size(0)):
            for j in range(lb.size(1)):
                for k in range(lb.size(2)):
                    if lb[i][j][k] >= 0.5:
                        lb[i][j][k] = 1.0
        '''
        #lb = lb[0]
        

        # load label
        #lb = int(self.train_labels[index])

        return im, lb

    def transform(self, img):
        # flip, crop, rotate
        p = np.random.rand(1)
        if p >= 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # rotate
        angle = np.random.uniform(-10, 10)
        img.rotate(angle)

        transform_img = transforms.Compose([transforms.Resize((22, 22)),
                                            transforms.ToTensor()])
        im = transform_img(img)
        return im
    
    def transform_1(self,img):
        transform_img = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            ])
        im = transform_img(img)
        return im
        
    def transform_2(self,img):
        transform_img = transforms.Compose([
                                            transforms.ToTensor(),
                                            ])
        im = transform_img(img)
        return im
    def transform_3(self,img):
        transform_img = transforms.Compose([transforms.Resize((256, 256),interpolation=0)
                                            
                                            ])
        im = transform_img(img)
        return im

name = 'E:\PYTHON\DUTS-TR\DUTS-TR\DUTS-TR-Mask\ILSVRC2012_test_00000018.png'

a = MyData()
q,w = a[10]
q1,q2,q3 = q[0],q[1],q[2]
print(w.shape)
q1 = np.array(q1)
q2 = np.array(q2)
q3 = np.array(q3)
w = np.array(w)
open('check_1.txt','w').write(str(q1.tolist()))
open('check_2.txt','w').write(str(w.tolist()))
'''
img = cv2.merge([w[0]*255])
cv2.imwrite('q.png',img)
img1 = cv2.merge([q1*255,q2*255,q3*255])
cv2.imwrite('w.png',img1)
'''
'''
# 显示图片
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyWindow('test')
'''
