# -*- coding: utf-8 -*-

from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
from imutils import paths
import random
random.seed(42)
class Demo:
    def __init__(self, model_path, le_path, detector_path, confidence):
        self.model_path = model_path
        self.detector_path = detector_path
        self.le_path = le_path
        self.confidence = confidence
        self.load_detection_model()

    def load_detection_model(self):
        print("[INFO] 加载人脸检测模型...")
        self.proto_path = os.path.sep.join([self.detector_path, "deploy.prototxt"])
        self.detector_model_path = os.path.sep.join([self.detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])

        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, caffeModel=self.detector_model_path)

    def start_detection(self):
        model = load_model(self.model_path)
        le = pickle.loads(open(self.le_path, "rb").read())
        print("[INFO] 正在启动摄像头...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0) #等待摄像头初始化
        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=600)

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            self.net.setInput(blob)
            detections = self.net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (227, 227))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]

                    label = "{}: {:.4f}".format(label, preds[j] - 0.03)
                    cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)

            cv2.imshow("LivenessDetection_Demo", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()

    def test_model(self, img_path):
        model = load_model(self.model_path)
        le = pickle.loads(open(self.le_path, "rb").read())

        img = cv2.imread(img_path)
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # face = img[startY:endY, startX:endX]
                face = cv2.resize(img, (227, 227))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le.classes_[j]

                label = "{}: {:.4f}".format(label, preds[j])
                print(img_path)
                print(label)

if __name__ == "__main__":
    demo = Demo('/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/final/liveness/liveness.model', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/final/liveness/train_le.pickle', '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/detector', 0.5)
    demo.start_detection()
    # demo.test_model('./test.png')
    # dataset_path = '../dataset/Homemade/real'
    # imgs = list(paths.list_images(dataset_path))
    # img = random.sample(imgs, 5)
    # for item in img:
    #     demo.test_model(item)
