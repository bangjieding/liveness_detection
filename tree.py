# -*- coding:utf-8 -*-
import os
import cv2
import random
import joblib
import pickle
import numpy as np
from math import ceil
from tqdm import tqdm
from imutils import paths
from random import sample
import skimage.feature as skif
from keras.utils import np_utils
from sklearn.svm import SVC, SVR
from sklearn.utils import shuffle
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

TRU = models.load_model(route_model)
CLU = joblib.load(classify_model)
TVU = [models.load_model(print_model), models.load(replay_model)]

