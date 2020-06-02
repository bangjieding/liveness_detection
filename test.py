import liveness_detection

# from imutils.video import VideoStream
# from keras.preprocessing.image import img_to_array
# from keras.models import load_model
# from keras.utils import np_utils
# import numpy as np
# import argparse
# import imutils
# import pickle
# import time
# import cv2
# import os
# from imutils import paths
# from sklearn.utils import shuffle
# import random
# from sklearn.metrics import classification_report
# from sklearn.preprocessing import LabelEncoder
# random.seed(42)


def get_imgs_data(img_list):
    imgs = []
    for img_path in img_list:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (227, 227))
        imgs.append(img)
    return np.array(imgs, dtype="float") / 255.0

dataset_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/NUAA_print'
real_train_txt_path = os.path.join(dataset_path, 'client_train_face.txt')
fake_train_txt_path = os.path.join(dataset_path, 'imposter_train_face.txt')
real_test_txt_path = os.path.join(dataset_path, 'client_test_face.txt')
fake_test_txt_path = os.path.join(dataset_path, 'imposter_test_face.txt')

real_train_img_path = []
real_test_img_path = []
fake_train_img_path = []
fake_test_img_path = []


for line in open(real_train_txt_path):
    img_path = line.strip('\n').split(' ')[0].replace('\\', os.path.sep)
    img_path = dataset_path + os.path.sep + 'ClientFace' + os.path.sep + img_path
    real_train_img_path.append(img_path)

for line in open(real_test_txt_path):
    img_path = line.strip('\n').split(' ')[0].replace('\\', os.path.sep)
    img_path = dataset_path + os.path.sep + 'ClientFace' + os.path.sep + img_path
    real_test_img_path.append(img_path)

for line in open(fake_train_txt_path):
    img_path = line.strip('\n').split(' ')[0].replace('\\', os.path.sep)
    img_path = dataset_path + os.path.sep + 'ImposterFace' + os.path.sep + img_path
    fake_train_img_path.append(img_path)

for line in open(fake_test_txt_path):
    img_path = line.strip('\n').split(' ')[0].replace('\\', os.path.sep)
    img_path = dataset_path + os.path.sep + 'ImposterFace' + os.path.sep + img_path
    fake_test_img_path.append(img_path)

with open(dataset_path + os.path.sep + 'real_train_img_path.txt', 'w+') as fw:
    for item in real_train_img_path:
        fw.write(item)
        fw.write('\n')

with open(dataset_path + os.path.sep + 'real_test_img_path.txt', 'w+') as fw:
    for item in real_test_img_path:
        fw.write(item)
        fw.write('\n')

with open(dataset_path + os.path.sep + 'fake_train_img_path.txt', 'w+') as fw:
    for item in fake_train_img_path:
        fw.write(item)
        fw.write('\n')

with open(dataset_path + os.path.sep + 'fake_test_img_path.txt', 'w+') as fw:
    for item in fake_test_img_path:
        fw.write(item)
        fw.write('\n')

# detector_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/detector'

# demo = liveness_detection.Demo('/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/output/AlexNet/Print/Print.model', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/output/AlexNet/Print/test_le.pickle', detector_path, 0.8)

# model = load_model('/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/output/AlexNet/Print/Print.model')
# le = pickle.loads(open('/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/output/AlexNet/Print/test_le.pickle', "rb").read())
X_test_temp = real_test_img_path + fake_test_img_path
y_test_temp = ['real'] * len(real_test_img_path)
y_test_temp = y_test_temp + ['fake'] * len(fake_test_img_path)
print(len(real_test_img_path))
print(len(fake_test_img_path))
print(len(real_train_img_path))
print(len(fake_train_img_path))

print(len(y_test_temp))
print(len(X_test_temp))
# X_test, y_test = shuffle(X_test_temp, y_test_temp, random_state=42)

# le_test = LabelEncoder()
# y_test = le_test.fit_transform(y_test)
# y_test = np_utils.to_categorical(y_test, len(le.classes_))
# data_test = get_imgs_data(X_test[:1000])
# predictions = model.predict(data_test, batch_size=50)
# print(classification_report(y_test[:1000].argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))