# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 21:36:51 2021

@author: 严天宇
"""

import torch
import torch.nn as nn
import copy
import cv2
import numpy as np


file = open('1.txt','r')
a = file.read()
a = eval(a)
a = a[0][0]
a = np.array(a)
# 图片的分辨率为200*300，这里b, g, r设为随机值，注意dtype属性
b = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
g = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
r = np.random.randint(0, 255, (224, 224), dtype=np.uint8)


# 合并通道，形成图片
img = cv2.merge([a[0]])
cv2.imwrite('1.png',img)

img = cv2.resize(img,(200,200))
# 显示图片
cv2.imshow('test', img)
cv2.waitKey(0)
cv2.destroyWindow('test')



