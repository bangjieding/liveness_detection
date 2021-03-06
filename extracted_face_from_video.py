# -*- coding: utf-8 -*-
from imutils.video import VideoStream
import numpy as np
import imutils
from imutils import paths
import time
import cv2
import os

class DataCollect:
    def __init__ (self, confidence, frame_num, detector_path='/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/detector'):
        self.confidence = confidence
        self.detector_path = detector_path
        self.frame_num = frame_num
        self.load_detection_model()

    def load_detection_model(self):
        print("[INFO] 加载人脸检测模型...")
        self.proto_path = os.path.sep.join([self.detector_path, "deploy.prototxt"])
        self.model_path = os.path.sep.join([self.detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, caffeModel=self.model_path)

    def collect_data_from_video(self, video_path):
        faces = []
        vc = cv2.VideoCapture(video_path)
        total_frame_num = vc.get(7)
        if total_frame_num < self.frame_num:
            skip = 1
        else:
            skip = total_frame_num // self.frame_num
        read = 0
        saved = 0
        while True:
            (grabbed, frame) = vc.read()
            if grabbed:
                read += 1
                if read % skip != 0:
                    continue
                elif saved != self.frame_num:
                    (h, w) = frame.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    self.net.setInput(blob)
                    detections = self.net.forward()
                    i = np.argmax(detections[0, 0, :, 2])
                    c = detections[0, 0, i, 2]
                    if c > self.confidence:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x_start, y_start, x_end, y_end) = box.astype(int)
                        face = frame[y_start:y_end, x_start:x_end]
                        faces.append(face)
                        saved += 1
                else:
                    break
            else:
                break
        vc.release()
        return faces