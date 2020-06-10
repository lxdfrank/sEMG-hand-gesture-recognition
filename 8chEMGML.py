#!/usr/bin/env python
# coding: utf-8
__author__ = 'Xiaodong Lv'

from keras.layers import Input, Dense, Dropout, BatchNormalization,\
                         LeakyReLU, concatenate,Conv2D, MaxPooling2D,\
                         AveragePooling2D, GlobalAveragePooling2D,Conv1D,\
                         Flatten,merge,Reshape,MaxPooling1D,Lambda,LSTM,\
                         GlobalMaxPooling2D
from keras import backend as K
from keras.layers.merge import Concatenate,add
from keras.models import Model
from keras.utils.vis_utils import plot_model
from scipy import signal
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np 
import scipy.io as scio
import copy
import os
import h5py
import numpy as np
import tensorflow as tf 
import keras

# 通过固定窗口进行滑窗

# myo手环的采样频率为200Hz
def rollWin(SINAL,Trigger,windows,step):
    if len(SINAL)%step == 0:
        total_STEP = int(len(SINAL)/step)
    else:
        total_STEP = int(len(SINAL)/step+1)

    Trigger = Trigger.T


    # the shape of arranged_ARRAY [16][100][n]
    arranged_ARRAY = np.zeros((8, windows, total_STEP),dtype=np.int)
    # the shape of arranged_ARRAY [n]
    arranged_LABEL = np.zeros(total_STEP)

    for x in range(total_STEP):
        if step*x+windows < len(SINAL):
            arranged_ARRAY[:,:,x] = SINAL.T[:,step*x:step*x+windows]
            arranged_LABEL[x] = int(np.median(Trigger[:,step*x:step*x+windows]))
        else:
            # arranged_ARRAY[:,0:((total_STEP-x)*step+len(SINAL)%step-1),x] = SINAL.T[:,(len(SINAL)-1)]
            arranged_ARRAY[:,0:((total_STEP-x)*step+len(SINAL)%step-1),x] = \
                                               SINAL.T[:,step*x:(len(SINAL)-1)]
            arranged_LABEL[x] = int(np.median(Trigger[:,step*x:(len(SINAL)-1)]))
            # 
            # arranged_LABEL[x] = int(np.argmax(np.bincount(array)))

    return arranged_ARRAY,arranged_LABEL

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# 滤波器设计
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def block_layer(x):
    x = Conv2D(64, (3, 3), strides=1, padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)

    x = Conv2D(12, (3, 3), strides=1, padding='same')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x

def connected_layer(x_input):

    conv1 = BatchNormalization(axis=3)(x_input)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(48, (1, 1), strides=1, padding='same')(conv1)
    conv1 = BatchNormalization(axis=3)(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(12, (1, 1), strides=1, padding='same')(conv1)
    conv1 = Dropout(0.2)(conv1)

    conv3 = BatchNormalization(axis=3)(x_input)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(48, (3, 3), strides=1, padding='same')(conv3)
    conv3 = BatchNormalization(axis=3)(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(12, (3, 3), strides=1, padding='same')(conv3)
    conv3 = Dropout(0.2)(conv3)

    conv5 = BatchNormalization(axis=3)(x_input)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(48, (5, 5), strides=1, padding='same')(conv5)
    conv5 = BatchNormalization(axis=3)(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(12, (5, 5), strides=1, padding='same')(conv5)
    conv5 = Dropout(0.2)(conv5)

    conv = concatenate([x_input,conv1,conv3,conv5], axis=3)

    return conv

def TransitionLayer(x_input, compression=0.5, alpha=0.0, is_max=1):
    
    nb_filter = int(x_input.shape.as_list()[-1]*compression)
    x_input = BatchNormalization(axis=3)(x_input)
    x_input = LeakyReLU(alpha=alpha)(x_input)
    x_input = Conv2D(nb_filter, (1, 1), strides=(1,1), padding='same')(x_input)
    x_input = AveragePooling2D(pool_size=(2, 2), strides=2)(x_input)

    return x_input

# 训练的模型是(16,30,1), classes=13
def NovalNet(input_shape=(8,30,1), classes=8):

    inpts = Input(input_shape)

    xx = Conv2D(24, (3, 3), strides=1, padding='same')(inpts)
    xx = BatchNormalization(axis=3)(xx)
    xx = LeakyReLU(alpha=0.1)(xx)

    # 创建CNN层进行特征提取
    for i in range(3):
        xx = connected_layer(xx)
    xx = TransitionLayer(xx)
    y = xx

    # y = GlobalAveragePooling2D()(y)

    y = GlobalMaxPooling2D()(y)


    y = Dropout(0.3)(y)

    # y = Flatten()(y)
    y = Dense(128)(y)
    y = Dense(128)(y)
    y = Dense(classes, activation='softmax', name='fc4')(y)

    model = Model(inputs=inpts, outputs=y, name='NovalNet')
    return model

def main():
    # DB2_Dir = 'F:\\Project\\Machinelearning\\MachineVision\\handGesture\\myodataset\\'
    # E1 = scio.loadmat(DB2_Dir+'myomat2.mat')
    # E1 = scio.loadmat(DB2_Dir+'myomat3.mat')
    # E1_emg = E1['emg']
    # E1_label = E1['stimulus'][0].T

    DB2_Dir = 'F:\\Project\\Machinelearning\\MachineVision\\handGesture\\myodataset\\'
    E1 = scio.loadmat(DB2_Dir+'myomat2.mat')
    E2 = scio.loadmat(DB2_Dir+'myomat3.mat')

    E1_emg = E1['emg']
    E1_label = E1['stimulus'][0].T

    E2_emg = E2['emg']
    E2_label = E2['stimulus'][0].T
    print('===============')
    print(E2_emg)
    print(E1_emg.shape,E2_emg.shape)

    E1_emg =  np.concatenate((E1_emg,E2_emg),axis=0)
    E1_label = np.concatenate((E1_label,E2_label),axis=0)
    
    E1_label = np.expand_dims(E1_label,axis=1)

    # 采样率 200Hz
    # 想要滤掉 0.1-90Hz
    fs = 200.0
    lowcut = 0.1
    highcut = 90.0
    fnq = fs*0.5
    low = lowcut/fnq
    high = highcut/fnq
    print(low,high)
    b,a = signal.butter(5,[low,high],btype='bandpass')
    for i in range(len(E1_emg.T)):
        E1_emg.T[i] = signal.filtfilt(b,a,E1_emg.T[i])

    rollWINDOWS = 30  
    rollSTEP = 10
    all_data, all_label = rollWin(E1_emg,E1_label,rollWINDOWS,rollSTEP)

    all_data = all_data.swapaxes(0,2)
    all_data = all_data.swapaxes(1,2)

    # 打乱数据生成打乱后的数据索引
    index = len(all_label)
    index = np.random.permutation(index)
    all_data = all_data[index,:,:]
    all_label = all_label[index].astype(np.int)
    all_data = np.expand_dims(all_data, axis=3)

    all_label = convert_to_one_hot(all_label,8).T

    # 划分数据集
    N = len(all_label)
    num_train = round(N*0.8)
    X_train = all_data[0:num_train,:,:,:]
    Y_train = all_label[0:num_train,:]
    X_test  = all_data[num_train:N,:,:,:]
    Y_test  = all_label[num_train:N,:]
    # X_train = all_data[0:num_train,:,:]
    # Y_train = all_label[0:num_train,:]
    # X_test  = all_data[num_train:N,:,:]
    # Y_test  = all_label[num_train:N,:]

    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    model = NovalNet(input_shape=(8,30,1),classes=8)

    # model.save('8chNovalNet.h5')
    model.summary()
    plot_model(model, to_file='8chNovalNet.png',show_shapes=True)

    import time
    start = time.time()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = LossHistory() # 创建一个history实例

    model.fit(X_train, Y_train, epochs=200, batch_size=128, verbose=1, 
                validation_data=(X_test, Y_test),callbacks=[history])
    preds_train = model.evaluate(X_train, Y_train)
    print("Train Loss = " + str(preds_train[0]))
    print("Train Accuracy = " + str(preds_train[1]))
    y_pred = model.predict(X_test,batch_size = 1)
    print(y_pred)
    print(Y_test)

    end = time.time()
    print("time:",end-start)
    history.loss_plot('epoch')

    # model.save('8chNovalNet.h5')
    # model.save_weights('8chNovalNetweight.h5')




if __name__ == '__main__':
    main()



    