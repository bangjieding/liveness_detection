# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.utils import plot_model

class AlexNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential() #模型初始化
        input_shape = (height, width, depth)
        chanDim = -1
        
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chanDim = 1
        #第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
        model.add(Conv2D(96,(11,11), strides=(4,4), input_shape=input_shape, padding='valid',activation='relu',kernel_initializer='uniform'))
        #使用池化层，步长为2
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
        model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        #使用池化层，步长为2
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        # 第三层卷积，大小为3x3的卷积核使用384个
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        # 第四层卷积,同第三层
        model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        # 第五层卷积使用的卷积核为256个，其他同上
        model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
        #使用池化层，步长为2
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        model.add(Flatten())
        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096,activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(classes,activation='softmax'))

        return model

# if __name__ == "__main__":
#     test = AlexNet()
#     model = test.build(227, 227, 3, 2)
#     model.summary()
#     plot_model(model, show_shapes=True, to_file='./output/model.ps')
