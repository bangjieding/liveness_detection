# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
np.set_printoptions(threshold = 1e6)
def arry_58():
    l = np.zeros(58, dtype=int)
    m = []
    a = -1
    for i in range(256):
        bit = '{:08b}'.format(i)  # 二进制化
        arry = []  # 二进制生成数组
        count = 0  # 计数变化次数
        # 把字符串变为数组方便统计变化次数
        for x in range(len(bit)):
            arry.append(int(bit[x]))
        # print(arry)
        first = arry[0]  # 数组第一个为first，与之后的比较
        for j in range(1, len(arry)):
            if arry[j] != first:  # 如果变化，计数单位加1
                count += 1
                first = arry[j]  # 并且把变化的值重新赋值
        # print(count)
        if count <= 2:  # 如果变化次数小于2，则依次排序
            a += 1
            # print(i)
            l[a] = i
    l = l.tolist()
    return l
def uniform_LBP(img):
    h, w = img.shape  # 图像的长和宽
    dst = np.zeros((h - 2, w - 2), dtype=img.dtype)  # 新建一张图
    # dst = np.zeros(img.shape, dtype=img.dtype)  # 新建一张图
    arry58 = arry_58()
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = img[i][j]
            code = []
            count = 0
 
            code7 = img[i - 1][j - 1]
            if code7 >= center:
                code7 = 1
            else:
                code7 = 0
            code.append(code7)
 
            code6 = img[i - 1][j]
            if code6 >= center:
                code6 = 1
            else:
                code6 = 0
            code.append(code6)
 
            code5 = img[i - 1][j + 1]
            if code5 >= center:
                code5 = 1
            else:
                code5 = 0
            code.append(code5)
 
            code4 = img[i][j + 1]
            if code4 >= center:
                code4 = 1
            else:
                code4 = 0
            code.append(code4)
 
            code3 = img[i + 1][j + 1]
            if code3 >= center:
                code3 = 1
            else:
                code3 = 0
            code.append(code3)
 
            code2 = img[i + 1][j]
            if code2 >= center:
                code2 = 1
            else:
                code2 = 0
            code.append(code2)
 
            code1 = img[i + 1][j - 1]
            if code1 >= center:
                code1 = 1
            else:
                code1 = 0
            code.append(code1)
 
            code0 = img[i][j - 1]
            if code0 >= center:
                code0 = 1
            else:
                code0 = 0
            code.append(code0)
            LBP = code7*128 + code6*64 + code5*32 + code4*16 + code3*8 + code2*4 + code1*2 + code0*1
            #print("LBP值为：",LBP)
            #print("8位编码为：",code)
            first = code[0]  # 数组第一个为first，与之后的比较
            for x in range(1, len(code)):
                if code[x] != first:  # 如果变化，计数单位加1
                    count += 1
                    first = code[x]  # 并且把变化的值重新赋值
            #print("跳变次数",count)
            if count > 2:  # 如果变化次数大于3，则归为59位
               dst[i - 1][j - 1] = 58
            else:  # 否则，按原来的数计算
                loca = arry58.index(LBP)  # 获取位置
                dst[i - 1][j - 1] = loca
    return dst
 
if __name__ == '__main__':
    gray = cv2.imread('./test.png', cv2.IMREAD_GRAYSCALE)
    print(gray.shape)
    a = uniform_LBP(gray)
    print(a.max())
    # cv2.imshow('uniform_lbp', a)
    cv2.imwrite("./2.jpg", a)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


    import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# 随机生成（10000,）服从正态分布的数据
data = np.random.randn(10000)
"""
绘制直方图
data:必选参数，绘图数据
bins:直方图的长条形数目，可选项，默认为10
normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
facecolor:长条形的颜色
edgecolor:长条形边框的颜色
alpha:透明度
"""
plt.hist(data, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
# 显示横轴标签
plt.xlabel("区间")
# 显示纵轴标签
plt.ylabel("频数/频率")
# 显示图标题
plt.title("频数/频率分布直方图")
plt.show()