# -*- coding: utf-8 -*-
from imutils.video import VideoStream
import numpy as np
import imutils
from imutils import paths
import time
import cv2
import os
from tqdm import tqdm
import extracted_face_from_video

class DataCollect:
    def __init__ (self, detector_path, confidence, skip):
        self.confidence = confidence
        self.detector_path = detector_path
        self.skip = skip
        self.load_detection_model()

    def load_detection_model(self):
        print("[INFO] 加载人脸检测模型...")
        self.proto_path = os.path.sep.join([self.detector_path, "deploy.prototxt"])
        self.model_path = os.path.sep.join([self.detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])

        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, caffeModel=self.model_path)

    def collect_data_from_image(self, input_path, output_path):
        print("[INFO] 从图像采集数据...")
        raw_img = cv2.imread(input_path)
        (h, w) = raw_img.shape[:2]
        saved = len(list(paths.list_images(output_path)))
        blob = cv2.dnn.blobFromImage(cv2.resize(raw_img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        i = np.argmax(detections[0, 0, :, 2])
        c = detections[0, 0, i, 2]
        if c > self.confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x_start, y_start, x_end, y_end) = box.astype(int)
            face = raw_img[y_start:y_end, x_start:x_end]
            p = os.path.sep.join([output_path, "{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1
        else:
            print("[ERROR]  未从图像中检测到人脸（{}）...".format(input_path))


    def collect_data_from_video(self, input_path, output_path):
        print("[INFO] 从视频采集数据...")
        vc = cv2.VideoCapture(input_path)

        read = 0
        saved = len(list(paths.list_images(output_path)))
        while True:
            (grabbed, frame) = vc.read()
            if grabbed:
                read += 1
                if read % self.skip != 0:
                    continue
                else:
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
                        p = os.path.sep.join([output_path, "{}.png".format(saved)])
                        cv2.imwrite(p, face)
                        saved += 1
                        # print("[INFO]{}已保存...".format(p))
            else:
                break
        print("[INFO]{}提取完毕...".format(input_path))
        vc.release()
        cv2.destroyAllWindows()
    
    def collect_data_from_camera(self, output_path):
        print("[INFO] 从摄像头采集数据...")
        read = 0
        saved = 0

        print("[INFO] 正在启动摄像头...")
        vs = VideoStream(src=0).start()
        # vs = cv2.VideoCapture(0)
        time.sleep(2.0) #等待摄像头初始化

        while True:
            frame = vs.read()
            frame = imutils.resize(frame, width=600)
            read += 1
            if read % self.skip != 0:
                continue
            else:
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
                    p = os.path.sep.join([output_path, "{}.png".format(saved)])
                    cv2.imwrite(p, face)
                    saved += 1
                    print("[INFO]{}已保存...".format(p))
            cv2.imshow("LivenessDetection_Demo(Replay)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cv2.destroyAllWindows()
        vs.stop()


if __name__ == "__main__":
    # dataset_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/CASIA_faceAntisp'
    # extract_face = extracted_face_from_video.DataCollect(0.5,   10)
    detector_path = './detector'
    output_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/test/real'
    video_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/dataset/Homemade/test_video/real/real.mov'
    demo = DataCollect(detector_path, 0.5, 1)
    demo.collect_data_from_video(video_path, output_path)
    # for line in open(dataset_path + os.path.sep + 'replay_attack.txt'):
    #     p = output_path + os.path.sep + 'replay'
    #     saved = len(list(paths.list_images(p)))
    #     img_list = extract_face.collect_data_from_video(line.strip('\n'))
    #     for img in img_list:
    #         cv2.imwrite(p + os.path.sep + str(saved) + '.png', img)
    #         saved = saved + 1
    
    # for line in open(dataset_path + os.path.sep + 'print_attack.txt'):
    #     p = output_path + os.path.sep + 'print'
    #     saved = len(list(paths.list_images(p)))
    #     img_list = extract_face.collect_data_from_video(line.strip('\n'))
    #     for img in img_list:
    #         cv2.imwrite(p + os.path.sep + str(saved) + '.png', img)
    #         saved = saved + 1

    # for line in open(dataset_path + os.path.sep + 'partial_attack.txt'):
    #     p = output_path + os.path.sep + 'partial'
    #     saved = len(list(paths.list_images(p)))
    #     img_list = extract_face.collect_data_from_video(line.strip('\n'))
    #     for img in img_list:
    #         cv2.imwrite(p + os.path.sep + str(saved) + '.png', img)
    #         saved = saved + 1
    
    # for line in open(dataset_path + os.path.sep + 'real_face.txt'):
    #     p = output_path + os.path.sep + 'real'
    #     saved = len(list(paths.list_images(p)))
    #     img_list = extract_face.collect_data_from_video(line.strip('\n'))
    #     for img in img_list:
    #         cv2.imwrite(p + os.path.sep + str(saved) + '.png', img)
    #         saved = saved + 1

    # print("总共{}张replay_attack.".format(len(list(paths.list_images(output_path + os.path.sep + 'replay')))))
    # print("总共{}张print_attack.".format(len(list(paths.list_images(output_path + os.path.sep + 'print')))))
    # print("总共{}张partial_attack.".format(len(list(paths.list_images(output_path + os.path.sep + 'partial')))))
    # print("总共{}张real.".format(len(list(paths.list_images(output_path + os.path.sep + 'real')))))
    # for line in open(dataset_path + os.path.sep + 'test.txt'):
    #     test_set.append(line.strip('\n'))
    
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

    # i = 0
    # for line in open(real_train_txt_path):
    #     i = i+1
    #     if i % 5 == 0:
    #         img_path = line.strip('\n').replace('\\', os.path.sep)
    #         img_path = dataset_path + os.path.sep + 'ClientRaw' + os.path.sep + img_path
    #         print(img_path)
    #         test.collect_data_from_image(img_path, '../dataset/Extracted/real')
    #         print('[INFO]{}'.format(img_path))
    # # i = 0
    # # for line in open(fake_train_txt_path):
    # #     i = i+1
    # #     if i % 10 == 0:
    # #         img_path = line.strip('\n').replace('\\', os.path.sep)
    # #         img_path = dataset_path + os.path.sep + 'ImposterRaw' + os.path.sep + img_path
    # #         test.collect_data_from_image(img_path, '../dataset/Extracted/print_NUAA')
    # #         print('[INFO]{}'.format(img_path))
    # i = 0
    # for line in open(real_test_txt_path):
    #     i = i+1
    #     if i % 5 == 0:
    #         img_path = line.strip('\n').replace('\\', os.path.sep)
    #         img_path = dataset_path + os.path.sep + 'ClientRaw' + os.path.sep + img_path
    #         test.collect_data_from_image(img_path, '../dataset/Extracted/real')
    #         print('[INFO]{}'.format(img_path))
    # i = 0
    # for line in open(fake_test_txt_path):
    #     i = i+1
    #     if i % 10 == 0:
    #         img_path = line.strip('\n').replace('\\', os.path.sep)
    #         img_path = dataset_path + os.path.sep + 'ImposterRaw' + os.path.sep + img_path
    #         test.collect_data_from_image(img_path, '../dataset/Extracted/print_NUAA')
    #         print('[INFO]{}'.format(img_path))