# -*- coding = utf-8 -*-
'''
Keras 的核心数据结构是 model，一种组织网络层的方式。最简单的模型是 Sequential 顺序模型，它由多个网络层线性堆叠
'''
from keras.models import Sequential

'''
批量规范化。该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1
'''
from keras.layers.normalization import BatchNormalization

'''
二维卷基层，即对图像的空间域卷积。该层对二维输入进行滑动窗卷积，当使用该层作为第一层时，应提供input_shape参数。例如input_shape = （128，128，3）代表128*128的彩色ＲＧＢ图像（data_format = ‘channels_last’）
'''
from keras.layers.convolutional import Conv2D 

'''
空间池化（也称为亚采样或下采样）降低了每个特征映射的维度，但是保留了最重要的信息，空间池可以有多种形式：最大（Max）,平均（Average），求和（Sum） 
以最大池化为例，我们定义了空间上的邻域（2x2的窗）并且从纠正特征映射中取出窗里最大的元素。除了取最大值以额外，我们也可以取平均值（平均池化）或者把窗里所有元素加起来。实际上，最大池化已经显示了最好的成效。
'''
from keras.layers.convolutional import MaxPooling2D

'''
对一个层的输出添加激活函数
'''
from keras.layers.core import Activation 

'''
Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
'''
from keras.layers.core import Flatten 

'''
为输入数据施加Dropout。Dropout将在训练过程中每次更新参数时随机断开一定百分比（rate）的输入神经元，Dropout层用于防止过拟合。
'''
from keras.layers.core import Dropout 

'''
用来对上一层的神经元进行全部连接，实现特征的非线性组合，就是常用的全连接层
'''
from keras.layers.core import Dense 

'''

'''
from keras import backend as K

class DetectionNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential() #模型初始化
        inputShape = (height, width, depth)
        chanDim = -1
        
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        '''
        Conv2D:
            keras.layers.convolutional.Conv2D(filters,kernel_size,strdes=(1,1),padding='valid',data_format=None,dilation_rate=(1,1),activation=None,use_bias=True,kernel_initialize='glorot_uniform',bias_initializer='zeros',kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,bias_constraint=None)
        '''
        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(16, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        # return the constructed network architecture
        return model