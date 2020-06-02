# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os

from detection_net import DetectionNet

from AlexNet import AlexNet
from AlexNetPro import AlexNetPro

import random
from random import sample
from math import ceil

random.seed(42)

class Config:
    INIT_LR = 1e-4
    BS = 50
    EPOCHS = 5

def split_train_vali_test(dataset_path, output_path):
    dataset = []
    for i in dataset_path:
        dataset = dataset + list(paths.list_images(i))
    train_set = []
    test_vali_set = []
    test_set = []
    vali_set = []

    dataset = shuffle(dataset, random_state=42)
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
    return np.array(imgs, dtype="float") / 255.0
    
def get_data_batch(img_set, labels, mode='invalid'):
    while True:
        aug = ImageDataGenerator(rotation_range=20, 
                             zoom_range=0.15,
                             width_shift_range=0.2, 
                             height_shift_range=0.2, 
                             shear_range=0.15, 
                             horizontal_flip=True, 
                             fill_mode="nearest")
        for i in range(0, len(img_set), Config.BS):
            X_ = get_imgs_data(img_set[i:i+Config.BS])
            y_ = labels[i:i+Config.BS]
            if mode=='train':
                aug.fit(X_)
                for X, y in aug.flow(X_, y_, batch_size=Config.BS):
                    yield(X, y)
            else:
                yield X_, y_

def compile_model(dataset_path, output_path, net):
    
    # (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    # (X_train, X_test_vali, y_train, y_test_vali) =  train_test_split(data, labels, test_size=0.20, random_state=42)
    # (X_vali, X_test, y_vali, y_test) = train_test_split(X_test_vali, y_test_vali, test_size=0.5, random_state=42)
    (X_train, X_vali, X_test, y_train, y_vali, y_test, le) = split_train_vali_test(dataset_path, output_path)
    
    print("[INFO] 编译模型...")
    opt = Adam(lr=Config.INIT_LR, decay=Config.INIT_LR / Config.EPOCHS)
    #model = DetectionNet.build(width=227, height=227, depth=3, classes=len(le.classes_))
    if net=='AlexNet':
        model = AlexNet.build(width=227, height=227, depth=3, classes=len(le.classes_))
    elif net=='AlexNetPro':
        model = AlexNetPro.build(width=227, height=227, depth=3, classes=len(le.classes_))

    if len(le.classes_) == 2:
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    else:
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    

    print("[INFO] 训练{} epochs...".format(Config.EPOCHS))
    # H = model.fit_generator(aug.flow(X_train, y_train, batch_size=Config.BS), validation_data=(X_vali, y_vali),steps_per_epoch=len(X_train) // Config.BS, epochs=Config.EPOCHS)
    
    H = model.fit_generator(generator=get_data_batch(X_train, y_train, mode='train'), 
                            validation_data=get_data_batch(X_vali, y_vali),
                            validation_steps=len(X_vali) // Config.BS,
                            steps_per_epoch=len(X_train) // Config.BS, 
                            epochs=Config.EPOCHS)

    model_path = os.path.join(output_path,'{}.model'.format(output_path.split(os.sep)[-1]))
    print("[INFO] 保存模型至{}".format(model_path))
    model.save(model_path)

    
    '''
    评价网络
    '''
    print("[INFO] 评价...")
    data_test = get_imgs_data(X_test)
    predictions = model.predict(data_test, batch_size=Config.BS)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))
    
 
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, Config.EPOCHS), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, Config.EPOCHS), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, Config.EPOCHS), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, Config.EPOCHS), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")

    evaluate_path = os.path.join(output_path, 'evaluate_{}.png'.format(output_path.split(os.sep)[-1]))
    print("[INFO] 保存评价至{}".format(evaluate_path))
    plt.savefig(evaluate_path)

def evaluate_model(model_path, le_path, X_test, y_test):
    le = pickle.loads(open(le_path, "rb").read())
    model = load_model(model_path)
    print("[INFO] 评价...")
    data_test = get_imgs_data(X_test)
    predictions = model.predict(data_test, batch_size=Config.BS)
    print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

if __name__ == "__main__":
    dataset_path = ['/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/test/fake', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/test/real']
    # (X_train, X_vali, X_test, y_train, y_vali, y_test, le) = split_train_vali_test(dataset_path, './output/lunwen')
    # model_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/final/CASIA/AlexNetPro/AlexNetPro.model'
    # le_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/final/CASIA/AlexNetPro/train_le.pickle'
    # evaluate_model(model_path, le_path, X_vali, y_vali)
    # dataset_path = ['../dataset/Extracted/print_NUAA', '../dataset/Extracted/real_NUAA']
    output_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/final/liveness'
    compile_model(dataset_path, output_path, 'AlexNetPro')
    # output_path = './final/NUAA_print/AlexNetPro'
    # compile_model(dataset_path, output_path, 'AlexNetPro')
    # dataset_path = ['../dataset/Extracted/print', '../dataset/Extracted/real','../dataset/Extracted/replay']
    # output_path = './final/CASIA/AlexNet'
    # compile_model(dataset_path, output_path, 'AlexNet')
    # output_path = './final/CASIA/AlexNetPro'
    # compile_model(dataset_path, output_path, 'AlexNetPro')

    # dataset_path = ['../dataset/Extracted/print', '../dataset/Extracted/real']
    # output_path = './output/AlexNetPro/Print'
    # compile_model(dataset_path, output_path, 'AlexNetPro')

    # dataset_path = ['../dataset/Extracted/replay', '../dataset/Extracted/real']
    # output_path = './output/AlexNetPro/Replay'
    # compile_model(dataset_path, output_path, 'AlexNetPro')

    # dataset_path = ['../dataset/Extracted/print', '../dataset/Extracted/replay', '../dataset/Extracted/real']
    # output_path = './output/AlexNet/Mix'
    # compile_model(dataset_path, output_path, 'AlexNet')

    # dataset_path = ['../dataset/Extracted/print', '../dataset/Extracted/real']
    # output_path = './output/AlexNet/Print'
    # compile_model(dataset_path, output_path, 'AlexNet')

    # dataset_path = ['../dataset/Extracted/replay', '../dataset/Extracted/real']
    # output_path = './output/AlexNet/Replay'
    # compile_model(dataset_path, output_path, 'AlexNet')
    

    # dataset_path = ['../dataset/Extracted/replay', '../dataset/Extracted/real']
    # output_path = './output/AlexNet/Replay'
    # compile_model(dataset_path, output_path, 'AlexNet')

    # print("----------------------Second-----------------------")
    # dataset_path = ['../dataset/Extracted/print', '../dataset/Extracted/real']
    # output_path = './output/AlexNet/Print'
    # compile_model(dataset_path, output_path, 'AlexNet')


    
    # dataset_path = ['../dataset/Homemade/fake', '../dataset/Homemade/real']
    # output_path = './output/test'
    # compile_model(dataset_path, output_path, 'AlexNetPro')