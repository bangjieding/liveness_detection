# -*- coding: utf-8 -*-
from random import sample
from math import ceil
import os

class Load:
    @staticmethod
    def get_train_batch(X_train, y_train, batch_size, img_w, img_h, color_type, is_argumentation):

   while 1:        

for i in range(0, len(X_train), batch_size):
           x = get_im_cv2(X_train[i:i+batch_size], img_w, img_h, color_type)
           y = y_train[i:i+batch_size]            

if is_argumentation:                

# 数据增强
               x, y = img_augmentation(x, y)            

# 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完

           yield({'input': x}, {'output': y})