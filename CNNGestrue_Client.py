import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate,Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D,Activation
from keras.models import load_model

import socket
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import regularizers
from keras.utils.vis_utils import plot_model

def readFrame():
    img = cv2.imread('E:\\Datasets\\GestureDatasettest\\1_6.jpg')
    # 尺寸比例缩放
    (x,y,_) = img.shape
    img = cv2.resize(img, (int(y / 10), int(x / 10)))
    _img = np.array(img).astype('float32') / 255.0
    return _img

model=load_model('handmodel.h5')
model.load_weights('handweight.h5')
# # model.summary()
# a = readFrame()
# a = np.array([a])
# print(a.shape)
# result=model.predict(a,batch_size=1)
# result = np.argmax(result, axis=1)
# print(result)


'''
读取摄像头
'''
isBgCaptured = 0
camera = cv2.VideoCapture(0)
'''
开启UDP
'''
address = ('127.0.0.1',31500)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(address)

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter\
    if isBgCaptured==1:
        # 背景减去
        fgmask = bgModel.apply(frame,learningRate=0)
        kernel = np.ones((5, 5), np.uint8)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        res = cv2.bitwise_and(frame, frame, mask=fgmask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)

        cv2.imshow('blur_origin', _blur)

        # 尺寸比例缩放
        (x,y,_) = _blur.shape
        _blur = cv2.resize(_blur, (int(y / 10), int(x / 10)))

        #数据归一化
        _img = np.array(_blur).astype('float32')/255.0
        _img = np.array([_img])
        result=model.predict(_img,batch_size=1)
        result = np.argmax(result, axis=1)
        print(result)
        s.sendto(result,address)

        cv2.imshow('blur', _blur)
    cv2.imshow('frame', frame)
    # Keyboard OP
    k = cv2.waitKey(10)
    if k == ord('z'):  # press ESC to exit
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(history=0, varThreshold=50, detectShadows=False)
        isBgCaptured = 1
        print( '!!!Background Captured!!!')