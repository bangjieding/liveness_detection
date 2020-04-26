from keras import layers
from keras import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, concatenate, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import plot_model

class AlexNetPro:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        channels_dim = -1
        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            channels_dim = 1
        
        img_input = Input(shape=input_shape, dtype='float32', name='img_input')

        level_1 = Conv2D(96, (11,11), strides=(4,4), input_shape=input_shape, padding='valid', activation='relu', kernel_initializer='uniform')(img_input)
        # level_1 = BatchNormalization(axis = channels_dim)(level_1)
        level_1_pooling = MaxPooling2D(pool_size=(3,3), strides=(2,2))(level_1)


        level_2_1x1 = Conv2D(32, (1,1), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform')(level_1_pooling)
        # level_2_1x1 = BatchNormalization(axis = channels_dim)(level_2_1x1)

        level_2_3x3 = Conv2D(96, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform')(level_1_pooling)
        # level_2_3x3 = BatchNormalization(axis = channels_dim)(level_2_3x3)

        level_2_5x5 = Conv2D(128, (5,5), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform')(level_1_pooling)
        # level_2_5x5 = BatchNormalization(axis = channels_dim)(level_2_5x5)

        level_2_con = concatenate([level_2_1x1, level_2_3x3, level_2_5x5], axis=channels_dim)
        level_2_pooling = MaxPooling2D(pool_size=(3,3), strides=(2,2))(level_2_con)


        level_3 = Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform')(level_2_pooling)
        # level_3 = BatchNormalization(axis=channels_dim)(level_3)


        level_4 = Conv2D(384, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform')(level_3)
        # level_4 = BatchNormalization(axis=channels_dim)(level_4)


        level_5 = Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu', kernel_initializer='uniform')(level_4)
        # level_5 = BatchNormalization(axis=channels_dim)(level_5)
        level_5_pooling = MaxPooling2D(pool_size=(3,3),  strides=(2,2))(level_5)
        flatten = Flatten()(level_5_pooling)


        fc_1 = Dense(4096, activation='relu')(flatten)
        fc_1 = Dropout(0.5)(fc_1)

        fc_2 = Dense(4096, activation='relu')(fc_1)
        fc_2 = Dropout(0.5)(fc_2)


        fake_type_or_real = Dense(classes, activation='softmax')(fc_2)


        model = Model(img_input, fake_type_or_real)

        return model



if __name__ == "__main__":
    test = AlexNetPro()
    model = test.build(227, 227, 3, 2)
    model.summary()
    plot_model(model, show_shapes=True, to_file='./output/modelpro.png')