# -*- coding:utf-8 -*-
import os
import cv2
import joblib
import numpy as np
from imutils import paths
import skimage.feature as skif
from sklearn.svm import SVC, SVR
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

RADIUS = 1
POINTS = 8
MODE = 'nri_uniform'

def load_dataset(dataset_paths):
    labels = []
    img_paths = []
    for path in dataset_paths:
        img_paths = img_paths + (list(paths.list_images(path)))
    img_paths = shuffle(img_paths, random_state=42)
    labels = [x.split(os.path.sep)[-2] for x in img_paths]
    return train_test_split(img_paths, labels, test_size=0.1, random_state=42)

def get_lbphs(img_paths):
    lbphs = []
    for img_path in img_paths:
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray = cv2.resize(gray, (300, 300))
        h, w = gray.shape
        block_lbphs = []
        for row in range(3):
            for col in range(3):
                img_block = gray[(row * h//3) : ((row+1) * h//3 - 1) , (col * w//3) : ((col+1) * w//3 - 1)]
                img_block_lbp = skif.local_binary_pattern(img_block, POINTS, RADIUS, MODE)
                img_block_lbp = img_block_lbp.astype(np.int32)
                max_bins = int(img_block_lbp.max() + 1)
                hist, _ = np.histogram(img_block_lbp, bins=max_bins, range=(0,max_bins))
                block_lbphs = block_lbphs + hist.tolist()
        lbphs.append(block_lbphs)
    return lbphs

def train_svc(X_train, y_train):
    clf = SVC(C=1, kernel='rbf', gamma='auto', decision_function_shape='ovo')
    print("[INFO] 开始训练... ")
    clf.fit(get_lbphs(X_train), y_train)
    joblib.dump(clf, "./output/lbp/lbp_clf.model")
    # print(clf.predict(get_lbphs(['./test.png', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/real/0.png', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/fake/0.png'])))

def test_svc(X_test, y_test='invalid'):
    clf = joblib.load("./output/lbp/lbp_clf.model")
    score = clf.score(get_lbphs(X_test), y_test)
    print(score)
    # print(clf.predict(get_lbphs(X_test)))

if __name__ == "__main__":
    dataset_paths = ['../dataset/Homemade/real', '../dataset/Homemade/fake']
    # dataset_paths = ['../dataset/Extracted/replay', '../dataset/Extracted/real']
    print("[INFO] 正在加载图片...")
    X_train, X_test, y_train, y_test = load_dataset(dataset_paths)
    print("[INFO] 训练集{}张图片，测试集{}张图片".format(len(X_train), len(X_test)))
    # get_lbphs(X_test)
    # print(X_test[0:10])
    # print(y_test[0:10])
    train_svc(X_train, y_train)
    test_svc(X_test, y_test)
    # test_svc(['./test.png', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Extracted/replay/0.png', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/fake/0.png'])
    # get_lbphs(X_test)