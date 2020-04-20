from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

import random
from random import sample
from math import ceil

random.seed(42)

def split_train_vali_test(dataset_path, output_path):
    dataset = []
    for i in dataset_path:
        dataset = dataset + list(paths.list_images(i))
    train_set = []
    test_vali_set = []
    test_set = []
    vali_set = []

    train_set = sample(dataset, ceil(len(dataset)*0.8))
    test_vali_set = set(dataset) - set(train_set)
    test_set = sample(test_vali_set, ceil(len(test_vali_set)*0.5))
    vali_set = set(test_vali_set) - set(test_set)
    vali_set = list(vali_set)
    print("[INFO] 训练集{}张图片，验证集{}张图片，测试集{}张图片".format(len(train_set), len(vali_set), len(test_set)))

    label_num = len(dataset_path)
    print('[INFO] 总共有{}类标签'.format(label_num))

    train_labels = [x.split(os.path.sep)[-2] for x in train_set]
    train_set, train_labels = shuffle(train_set, train_labels, random_state=42)
    le_train = LabelEncoder()
    train_labels = le_train.fit_transform(train_labels)
    train_labels = np_utils.to_categorical(train_labels, label_num)
    train_output_path = os.path.join(output_path, 'train_le.pickle')
    print("[INFO] 保存标签至{}".format(train_output_path))
    f = open(train_output_path, "wb")
    f.write(pickle.dumps(le_train))
    f.close()

    vali_labels = [x.split(os.path.sep)[-2] for x in vali_set]
    vali_set, vali_labels = shuffle(vali_set, vali_labels, random_state=42)
    le_vali = LabelEncoder()
    vali_labels = le_vali.fit_transform(vali_labels)
    vali_labels = np_utils.to_categorical(vali_labels, label_num)
    vali_output_path = os.path.join(output_path, 'vali_le.pickle')
    print("[INFO] 保存标签至{}".format(vali_output_path))
    f = open(vali_output_path, "wb")
    f.write(pickle.dumps(le_vali))
    f.close()

    test_labels = [x.split(os.path.sep)[-2] for x in test_set]
    test_set, test_labels = shuffle(test_set, test_labels, random_state=42)
    le_test = LabelEncoder()
    test_labels = le_test.fit_transform(test_labels)
    test_labels = np_utils.to_categorical(test_labels, label_num)
    test_output_path = os.path.join(output_path, 'test_le.pickle')
    print("[INFO] 保存标签至{}".format(test_output_path))
    f = open(test_output_path, "wb")
    f.write(pickle.dumps(le_test))
    f.close()

    return train_set, vali_set, test_set, train_labels, vali_labels, test_labels, le_train

def get_imgs_data(img_list):
    imgs = []
    for img_path in img_list:
        img = cv2.imread(img_path)
        img =cv2.resize(img, (227, 227))
        imgs.append(img)
    return np.array(imgs, dtype="float")
    
def get_data_batch(img_set, labels, mode='invalid'):
    aug = ImageDataGenerator(rotation_range=20, 
                            zoom_range=0.15,
                            width_shift_range=0.2, 
                            height_shift_range=0.2, 
                            shear_range=0.15, 
                            horizontal_flip=True, 
                            fill_mode="nearest")
    for i in range(0, len(img_set), 10):
        X_ = get_imgs_data(img_set[i:i+10])
        y_ = labels[i:i+10]
        if mode=='train':
            aug.fit(X_)
            for X, y in aug.flow(X_, y_, batch_size=10):
                yield(X, y)
            # yield img_set[i:i+Config.BS]
        else:
            yield X_, y_

if __name__ == "__main__":
    dataset_path = ['../dataset/Homemade/fake', '../dataset/Homemade/real']
    output_path = './output/test'
    (X_train, X_vali, X_test, y_train, y_vali, y_test, le) = split_train_vali_test(dataset_path, output_path)
    i = 0
    for item in get_data_batch(X_test, y_test, mode='train'):
        print(i)
        for img in item[0]:
            cv2.imwrite('./output/test/img/{}'.format(str(len(list(paths.list_images('./output/test/img'))))+'.jpg'), img)