# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

RADIUS = 1
POINTS = 8
MODE = 'nri_uniform'

matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
def uniform_LBP(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    img_lbp = local_binary_pattern(img, POINTS, RADIUS, MODE)
    cv2.imwrite('./zhijie.png', img_lbp)
    img_lbp = img_lbp.astype(np.int32)
    max_bins = int(img_lbp.max() + 1)
    img_lbph = np.histogram(img_lbp, normed=True, bins=max_bins, range=(0, max_bins))

    return img_lbph

if __name__ == "__main__":
    lbph = []
    img_path = './test.png'
    lbph = lbph + uniform_LBP(img_path)
    