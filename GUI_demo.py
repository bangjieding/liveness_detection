# -*- coding:utf-8 -*-
# # import cv2
# # import tkinter as tk
# # from tkinter import filedialog#文件控件
# # from PIL import Image, ImageTk#图像控件
# # import threading#多线程
# # from imutils.video import VideoStream

# # #---------------打开摄像头获取图片
# # def video_demo():
# #     def cc():
# #         print('start')
# #         # capture = cv2.VideoCapture(0)
# #         capture = VideoStream(src=0).start()
# #         while True:
# #             ret, frame = capture.read()#从摄像头读取照片
# #             frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
# #             cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
# #             img = Image.fromarray(cv2image)
# #             mage_file=ImageTk.PhotoImage(img)
# #             canvas.create_image(0,0,anchor='nw',image=image_file)
# #             t=threading.Thread(target=cc)
# #             t.start()
# # #---------------创建窗口
# # window = tk.Tk()
# # window.title('摄像头')
# # sw = window.winfo_screenwidth()#获取屏幕宽
# # sh = window.winfo_screenheight()#获取屏幕高
# # wx = 600
# # wh = 800
# # window.geometry("%dx%d+%d+%d" %(wx,wh,(sw-wx)/2,(sh-wh)/2-100))#窗口至指定位置
# # canvas = tk.Canvas(window,bg='#c4c2c2',height=wh,width=wx)#绘制画布
# # canvas.pack()
# # bt_start = tk.Button(window,text='打开摄像头',height=2,width=15,command=video_demo)
# # bt_start.place(x=230,y=600)
# # window.mainloop()


# from tkinter import *
# import cv2
# from PIL import Image,ImageTk
# from imutils.video import VideoStream

# def take_snapshot():
#     print("有人给你点赞啦！")

# def video_loop():
#     img = camera.read()  # 从摄像头读取照片

#     cv2.waitKey(1000)
#     cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
#     current_image = Image.fromarray(cv2image)#将图像转换成Image对象
#     imgtk = ImageTk.PhotoImage(image=current_image)
#     panel.imgtk = imgtk
#     panel.config(image=imgtk)
#     root.after(1, video_loop)


# camera = VideoStream(src=0).start()    #摄像头

# root = Tk()
# root.title("opencv + tkinter")
# #root.protocol('WM_DELETE_WINDOW', detector)

# panel = Label(root)  # initialize image panel
# panel.pack(padx=10, pady=10)
# root.config(cursor="arrow")
# btn = Button(root, text="点赞!", command=take_snapshot)
# btn.pack(fill="both", expand=True, padx=10, pady=10)

# video_loop()

# root.mainloop()
# # 当一切都完成后，关闭摄像头并释放所占资源
# camera.release()
# cv2.destroyAllWindows()

import tkinter as tk
from PIL import Image,ImageTk
from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import time
import cv2
import os
from imutils import paths

class APP:
    def __init__(self, window):
        self.root = window
        self.model_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/output/AlexNetPro/Mix/Mix.model'
        self.le = pickle.loads(open('/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/output/AlexNetPro/Mix/train_le.pickle', "rb").read())
        self.confidence = 0.5
        self.detector_path = '/Users/DingBangjie/Documents/Tintin/Study/Graduate/code/liveness_detection/detector'
        self.load_detection_model()
       
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.init_window()
        self.call_camera()
    
    def load_detection_model(self):
        self.model = load_model(self.model_path)
        print("[INFO] 加载人脸检测模型...")
        self.proto_path = os.path.sep.join([self.detector_path, "deploy.prototxt"])
        self.detector_model_path = os.path.sep.join([self.detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
        self.net = cv2.dnn.readNetFromCaffe(self.proto_path, caffeModel=self.detector_model_path)
    
    def init_window(self):
        self.left_frame = tk.Frame(self.root)
        self.right_frame = tk.Frame(self.root)
        self.left_frame.rowconfigure(0, weight=1)
        self.left_frame.columnconfigure(0, weight=1)
        self.right_frame.rowconfigure(0, weight=1)
        self.right_frame.columnconfigure(1, weight=10)
        self.left_frame.pack(fill='both', expand='yes', side='left', padx='10', pady='50')
        self.right_frame.pack(fill='both', expand='yes', side='left', pady='40')

        self.frame_top = tk.Frame(self.left_frame)
        self.frame_top.pack(fill='both',expand='yes')
        self.frame_1 = tk.Frame(self.left_frame)
        self.frame_1.pack(fill='both',expand='yes')
        self.frame_2 = tk.Frame(self.left_frame)
        self.frame_2.pack(fill='both',expand='yes')
        self.frame_3 = tk.Frame(self.left_frame)
        self.frame_3.pack(fill='both',expand='yes')
        self.frame_4 = tk.Frame(self.left_frame)
        self.frame_4.pack(fill='both',expand='yes')

        self.top_label1 = tk.Label(self.frame_top, text='FACE',)
        self.top_label1.pack(side='left',expand='yes')
        self.top_label2 = tk.Label(self.frame_top, text='TYPE')
        self.top_label2.pack(side='left',expand='yes')
        self.top_label3 = tk.Label(self.frame_top, text='CONFIDENCE')
        self.top_label3.pack(side='left',expand='yes')

        self.face1 = tk.Label(self.frame_1,bg='red')
        self.face1.pack(side='left', expand='yes')
        self.face1_type = tk.StringVar()
        self.face1_confidence = tk.StringVar()
        self.face1_label1 = tk.Label(self.frame_1, textvariable=self.face1_type)
        self.face1_label1.pack(side='left', expand='yes')
        self.face1_label2 = tk.Label(self.frame_1, textvariable=self.face1_confidence)
        self.face1_label2.pack(side='left', expand='yes')

        self.face2 = tk.Label(self.frame_2,bg='red')
        self.face2.pack(side='left', expand='yes')
        self.face2_type = tk.StringVar()
        self.face2_confidence = tk.StringVar()
        self.face2_label1 = tk.Label(self.frame_2, textvariable=self.face2_type)
        self.face2_label1.pack(side='left', expand='yes')
        self.face2_label2 = tk.Label(self.frame_2, textvariable=self.face2_confidence)
        self.face2_label2.pack(side='left', expand='yes')

        self.face3 = tk.Label(self.frame_3,bg='red')
        self.face3.pack(side='left', expand='yes')
        self.face3_type = tk.StringVar()
        self.face3_confidence = tk.StringVar()
        self.face3_label1 = tk.Label(self.frame_3, textvariable=self.face3_type)
        self.face3_label1.pack(side='left', expand='yes')
        self.face3_label2 = tk.Label(self.frame_3, textvariable=self.face3_confidence)
        self.face3_label2.pack(side='left', expand='yes')

        self.face4 = tk.Label(self.frame_4,bg='red')
        self.face4.pack(side='left', expand='yes')
        self.face4_type = tk.StringVar()
        self.face4_confidence = tk.StringVar()
        self.face4_label1 = tk.Label(self.frame_4, textvariable=self.face4_type)
        self.face4_label1.pack(side='left', expand='yes')
        self.face4_label2 = tk.Label(self.frame_4, textvariable=self.face4_confidence)
        self.face4_label2.pack(side='left', expand='yes')
        
        self.camera =  VideoStream(src=0).start()
        self.camera_label = tk.Label(self.right_frame)
        self.camera_label.pack(fill='both', expand='yes')
        
    
    def call_camera(self):
        img = self.camera.read()
        img = cv2.resize(img, (480,270))
        self.start_detection(img)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
        current_image = Image.fromarray(cv2image)#将图像转换成Image对象
        imgtk = ImageTk.PhotoImage(image=current_image)
        self.camera_label.imgtk = imgtk
        self.camera_label.config(image=imgtk)
        self.root.after(1, self.call_camera)

    def start_detection(self, img):
        img = imutils.resize(img, width=600)
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        self.net.setInput(blob)
        detections = self.net.forward()
        for i in range(0, detections.shape[2]):
            if i==0:
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = img[startY:endY, startX:endX]
                    
                    # detected_face = face
                    detected_face = cv2.resize(face, (90, 140))
                    cv2image = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
                    current_image = Image.fromarray(cv2image)#将图像转换成Image对象
                    imgtk = ImageTk.PhotoImage(image=current_image)
                    self.face1.imgtk = imgtk
                    self.face1.config(image=imgtk)

                    face = cv2.resize(face, (227, 227))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = self.model.predict(face)[0]
                    j = np.argmax(preds)
                    label = self.le.classes_[j]
                    confi = preds[j]
                    
                    self.face1_type.set(label)
                    self.face1_confidence.set(confi)

            if i==1:
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = img[startY:endY, startX:endX]
                    
                    # detected_face = face
                    detected_face = cv2.resize(face, (90, 140))
                    cv2image = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
                    current_image = Image.fromarray(cv2image)#将图像转换成Image对象
                    imgtk = ImageTk.PhotoImage(image=current_image)
                    self.face2.imgtk = imgtk
                    self.face2.config(image=imgtk)

                    face = cv2.resize(face, (227, 227))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = self.model.predict(face)[0]
                    j = np.argmax(preds)
                    label = self.le.classes_[j]
                    confi = preds[j]
                    
                    self.face2_type.set(label)
                    self.face2_confidence.set(confi)

            if i==2:
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = img[startY:endY, startX:endX]
                    
                    # detected_face = face
                    detected_face = cv2.resize(face, (90, 140))
                    cv2image = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
                    current_image = Image.fromarray(cv2image)#将图像转换成Image对象
                    imgtk = ImageTk.PhotoImage(image=current_image)
                    self.face3.imgtk = imgtk
                    self.face3.config(image=imgtk)

                    face = cv2.resize(face, (227, 227))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = self.model.predict(face)[0]
                    j = np.argmax(preds)
                    label = self.le.classes_[j]
                    confi = preds[j]
                    
                    self.face3_type.set(label)
                    self.face3_confidence.set(confi)
            
            if i==3:
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = img[startY:endY, startX:endX]
                    
                    # detected_face = face
                    detected_face = cv2.resize(face, (90, 140))
                    cv2image = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGBA)#转换颜色从BGR到RGBA
                    current_image = Image.fromarray(cv2image)#将图像转换成Image对象
                    imgtk = ImageTk.PhotoImage(image=current_image)
                    self.face4.imgtk = imgtk
                    self.face4.config(image=imgtk)

                    face = cv2.resize(face, (227, 227))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = self.model.predict(face)[0]
                    j = np.argmax(preds)
                    label = self.le.classes_[j]
                    confi = preds[j]
                    
                    self.face4_type.set(label)
                    self.face4_confidence.set(confi)

window = tk.Tk()
window.title('LivenessDetection & SpoofingType')
sw = window.winfo_screenwidth()#获取屏幕宽
sh = window.winfo_screenheight()#获取屏幕高
ww = 900
wh = 700
x = (sw-ww)/2
y = (sh-wh)/2
window.geometry("%dx%d+%d+%d" %(ww, wh, x, y))#窗口至指定位置    
app = APP(window)
window.mainloop()