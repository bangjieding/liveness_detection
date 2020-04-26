# -*- coding:utf-8 -*-
from keras.models import load_model
from train import get_imgs_data
from imutils import paths
import os 
from sklearn.utils import shuffle
import random
from random import sample
from math import ceil
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.utils import np_utils
random.seed(42)

model_path = './output/AlexNetPro/Mix/Mix.model'
le_path =  './output/AlexNetPro/Replay/train_le.pickle'

# test_set = ['/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Extracted/print_NUAA','/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Extracted/real_NUAA' ]

test_set = ['/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/replay', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/real']

img_list =[]

img_list = img_list + list(paths.list_images(test_set[0]))

# real_list = list(paths.list_images(test_set[1]))
# temp_list = sample(shuffle(real_list, random_state=42), ceil(len(real_list)*0.1))

img_list = img_list + list(paths.list_images(test_set[1]))

img_list = shuffle(img_list, random_state=42)

print(len(img_list))
y_test = [x.split(os.path.sep)[-2] for x in img_list]
le_test = LabelEncoder()
y_test = le_test.fit_transform(y_test)
y_test = np_utils.to_categorical(y_test, 3)

X_test = get_imgs_data(img_list)

model = load_model(model_path)
le = pickle.loads(open(le_path, "rb").read())

predictions = model.predict(X_test, batch_size=50)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))