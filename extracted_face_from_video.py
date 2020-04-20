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
        print("[INFO] 从视频采集数据...")
        vc = cv2.VideoCapture(video_path)
        total_frame_num = vc.get(7)
        self.skip = total_frame_num // self.frame_num
        read = 0
        saved = 0
        while True:
            (grabbed, frame) = vc.read()
            if grabbed:
                read += 1
                if read % self.skip != 0:
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
        print(len(faces))
        print(type(faces[0]))
        vc.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = './videos/fake.mov'
    test = DataCollect(0.5, 50)
    test.collect_data_from_video(video_path)

    # dataset_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/CASIA_faceAntisp'
    # # video_path = [os.path.join(dataset_path, path) for path in ['train', 'test']]
    # train_path = [os.path.join(dataset_path, path) for path in ['train' + os.path.sep + str(x) for x in list(range(1, 21))]]
    # test_path = [os.path.join(dataset_path, path) for path in ['test' + os.path.sep + str(x) for x in list(range(1, 31))]]
    # train_video_path = []
    # test_video_path = []
    # for path in train_path:
    #     train_video_path.extend(os.path.join(path, subpath)  for subpath in ['7.avi', '8.avi']) 
    
    # for path in test_path:
    #     test_video_path.extend(os.path.join(path, subpath)  for subpath in ['7.avi', '8.avi'])

    # # print(train_video_path)
    # # print(test_video_path)

    # output_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/replay'
    # detector_path = './detector'
    # test = DataCollect(detector_path, 0.5, 10)
    # # test.collect_data_from_video(train_video_path[0], output_path)
    # # train_video_path = train_video_path[0:5]
    # for path in train_video_path:
    #     test.collect_data_from_video(path, output_path)

    # for path in test_video_path:
    #     test.collect_data_from_video(path, output_path)
    
    # print('[INFO] 总共提取了{}张replay图像。'.format(len(list(paths.list_images(output_path)))))
    # detector_path = './detector'
    # dataset_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/raw'
    # test = DataCollect(detector_path, 0.5, 10)

    # real_train_txt_path = os.path.join(dataset_path, 'client_train_raw.txt')
    # fake_train_txt_path = os.path.join(dataset_path, 'imposter_train_raw.txt')

    # real_test_txt_path = os.path.join(dataset_path, 'client_test_raw.txt')
    # fake_test_txt_path = os.path.join(dataset_path, 'imposter_test_raw.txt')


    # for line in open(real_train_txt_path):
    #     img_path = line.strip('\n').replace('\\', os.path.sep)
    #     img_path = dataset_path + os.path.sep + 'ClientRaw' + os.path.sep + img_path
    #     print(img_path)
    #     test.collect_data_from_image(img_path, '../dataset/real')
    #     print('[INFO]{}'.format(img_path))
        
    
    # for line in open(fake_train_txt_path):
    #     img_path = line.strip('\n').replace('\\', os.path.sep)
    #     img_path = dataset_path + os.path.sep + 'ImposterRaw' + os.path.sep + img_path
    #     test.collect_data_from_image(img_path, '../dataset/print')
    #     print('[INFO]{}'.format(img_path))
    
    # for line in open(real_test_txt_path):
    #     img_path = line.strip('\n').replace('\\', os.path.sep)
    #     img_path = dataset_path + os.path.sep + 'ClientRaw' + os.path.sep + img_path
    #     test.collect_data_from_image(img_path, '../dataset/real')
    #     print('[INFO]{}'.format(img_path))

    # for line in open(fake_test_txt_path):
    #     img_path = line.strip('\n').replace('\\', os.path.sep)
    #     img_path = dataset_path + os.path.sep + 'ImposterRaw' + os.path.sep + img_path
    #     test.collect_data_from_image(img_path, '../dataset/print')
    #     print('[INFO]{}'.format(img_path))