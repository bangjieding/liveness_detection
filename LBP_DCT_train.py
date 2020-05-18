# -*- coding:utf-8 -*-
import os
import cv2
import joblib
import numpy as np
from tqdm import tqdm
from imutils import paths
import skimage.feature as skif
from sklearn.svm import SVC, SVR
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from extracted_face_from_video import DataCollect

RADIUS = 1
POINTS = 8
MODE = 'nri_uniform'

def load_dataset(dataset_paths, enlabel=False):
    labels = []
    img_paths = []
    for path in dataset_paths:
        img_paths.append(list(paths.list_images(path)))
    if enlabel:
        labels.append(img_paths[0][0].split(os.path.sep)[-2])
        labels.append(img_paths[1][0].split(os.path.sep)[-2])
        return img_paths, labels
    else:
        return img_paths

def get_lbphs(img_list):
    lbphs = []
    for img in img_list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.resize(gray, (300, 300))
        #存储一帧图像的分块LBP编码
        # h, w = gray.shape
        # block_lbphs = [] 
        # for row in range(3):
        #     for col in range(3):
        #         img_block = gray[(row * h//3) : ((row+1) * h//3 - 1) , (col * w//3) : ((col+1) * w//3 - 1)]
        #         img_block_lbp = skif.local_binary_pattern(img_block, POINTS, RADIUS, MODE)
        #         max_bins = int(img_block_lbp.max() + 1)
        #         hist, _ = np.histogram(img_block_lbp, bins=max_bins, range=(0, max_bins))
        #         block_lbphs = block_lbphs + hist.tolist()
        # lbphs.append(block_lbphs)

        # 以下是不分块图片LBP提取
        img_uni_lbp = skif.local_binary_pattern(gray, POINTS, RADIUS, MODE)
        img_uni_lbp = img_uni_lbp.astype(np.int32)
        max_bins = int(img_uni_lbp.max() + 1)
        hist, _ = np.histogram(img_uni_lbp, bins=max_bins, range=(0,max_bins))
        lbphs.append(hist.tolist())
        # lbphs = lbphs + hist.tolist()
    return lbphs




def get_dct(videoset):
    extract_face = DataCollect(0.8, 40)
    X_dct_lpf = []
    X_dct     = []
    X_lbph    = []
    for video_path in tqdm(videoset):
        img_list = extract_face.collect_data_from_video(video_path)
        lbphs = get_lbphs(img_list)
        X_lbph.append(lbphs)
        lbphs = np.array(lbphs).astype(np.float32)
        lbphs_dct_lpf = []
        lbphs_dct = []
        for i in range(lbphs.shape[0]):
            lbph_dct = cv2.dct(lbphs[i,:])
            lbphs_dct = lbphs_dct + lbph_dct.T[0].tolist()
            lbph_dct_lpf = lbph_dct.T.take([0,1,2,3,4,5,6,7], axis=1)[0].tolist()
            lbphs_dct_lpf = lbphs_dct_lpf + lbph_dct_lpf
        X_dct.append(lbphs_dct)
        X_dct_lpf.append(lbphs_dct_lpf)
    print(np.array(X_dct_lpf).shape)
    return X_lbph, X_dct, X_dct_lpf

def get_minibatch(data, labels, batch_size):
    while True:
        for i in range(0, size(dataset), batch_size):
            X = get_lbphs(dataset[i:i+batch_size])
            y = labels[i:i+batch_size]
            yield (X, y)

def train_svc(X_train, y_train):
    clf = SVC(C=1e4, kernel='rbf', gamma='auto', decision_function_shape='ovo')
    print("[INFO] 开始训练... ")
    clf.fit(get_dct(X_train)[1], y_train)
    joblib.dump(clf, "/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/output/lbp/lbp+dct_clf.model")
    # print(clf.predict(get_lbphs(['./test.png', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/real/0.png', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/fake/0.png'])))

def test_svc(X_test, y_test='invalid'):
    clf = joblib.load("/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/output/lbp/lbp+dct_clf.model")
    print(clf.score(get_dct(X_test)[1], y_test))
    # print(clf.predict(get_dct(X_test)))

if __name__ == "__main__":
    dataset_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/CASIA_faceAntisp'
    train_set = []
    test_set = []
    for line in open(dataset_path + os.path.sep + 'train.txt'):
        train_set.append(line.strip('\n'))
    for line in open(dataset_path + os.path.sep + 'test.txt'):
        test_set.append(line.strip('\n'))
    train_labels = [x.split(os.path.sep)[-1].strip('.avi').replace('1', 'real').replace('7', 'replay') for x in train_set]
    test_labels = [x.split(os.path.sep)[-1].strip('.avi').replace('1', 'real').replace('7', 'replay') for x in test_set]
    # print(train_set[0:10])
    # print(train_labels[0:10])
    # get_dct(train_set[0:3])
    # train_svc(train_set, train_labels)
    test_svc(test_set[0:5], test_labels[0:5])
