# -*- coding: utf-8 -*-
"""
Created on Wed May 26 15:20:56 2021

@author: 严天宇
"""

from PIL import Image
from torchvision import transforms

image = Image.open("photo.jpg")
image = image.transpose(Image.ROTATE_270)
#image.transpose(Image.FLIP_LEFT_RIGHT)
#image.transpose(Image.FLIP_TOP_BOTTOM)
box = (0, 0, 500, 600)
image = image.crop(box)
image.show()