import numpy as np
from numpy import *
import keras
import tensorflow as tf
import cv2
from tqdm import tqdm
import os
from Constant import *
import pickle
import gc
import keras.applications.vgg16
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose, BatchNormalization, Activation
import h5py
#import pickle
from keras.models import load_model
from keras.utils import CustomObjectScope
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import scipy.misc
from scipy.misc.pilutil import imsave 
from Constant import *


def main(inp , out):
    vgg16 = keras.applications.vgg16.VGG16()
    vgg16.layers.pop()
    vgg16.layers.pop()
    vgg16.layers.pop()
    vgg16.layers.pop()
    vgg16.layers.pop()


    model = Sequential()
    for layer in vgg16.layers:
        model.add(layer)


    model.add(BatchNormalization())


    model.summary()

    #file = open(DATASET_INDEX)
    input =inp

    #INDEX = np.loadtxt(file, delimiter=',', dtype='str')
    INDEX = os.listdir(inp)
    DSal_weighted = os.path.join(Model_DIR, '3DSal-weighted.hdf5')
    modelsal = load_model(DSal_weighted)

"""
    This method creates the numpy training data

    from the images dataset, for each video individualy

    """


def create_training_data():
    print('Converting Training video images to numpy array ...')
    for index in INDEX:
        index=str(index)
        print(index)
        os.mkdir(input+'/'+index)
        path_train = TR_IMG_DIR + index + '/'
        # path_train = os.path.join(TR_IMG_DIR1, index)
        training_data = []

        for img in tqdm(os.listdir(path_train)):
            if img =='saliency':
              continue 
            img_array = cv2.imread(os.path.join(path_train,img), cv2.COLOR_RGB2BGR)

            new_array = cv2.resize(img_array, IMG_SIZE).astype(float)
            # im = np.expand_dims(new_array, axis=0)
            # a = model.predict(im)
            training_data.append([new_array])

        training_data = np.array(training_data).reshape(-1, 224, 224, 3)
        s = feature_map_function(training_data)
        del training_data
        X = Batch_Creation(s)
        X = np.array(X)
        del s
        for i in range(0, len(X), 1):
            Predictions = modelsal.predict(X[i:i + 1, :, :, :, :])
            p = Predictions[0][0]
            s = np.array(p[:, :, 0])
            imsave(input + index + '/' + str(format(i + 1, '04')) + '.jpg', s)
            print(i)

    gc.collect()
    print('Converting Training video images to numpy array ok ...')



def feature_map_function(X):

        m = []

        for i in range(len(X)):
            n = model.predict(X[i:i+1,:,:,:])
            k = n[0]
            m.append(k)

        return m


def Batch_Creation(X):

        K = []
        S = []
        for i in range(len(X)):
            if   i==0:
                K = np.array([X[i], X[i], X[i], X[i], X[i], X[i]])
            elif i==1:
                K = np.array([X[i-1], X[i-1], X[i], X[i], X[i], X[i]])
            elif i==2:
                K = np.array([X[i-2], X[i-1], X[i-1], X[i], X[i], X[i]])
            elif i==3:
                K = np.array([X[i-3], X[i-2], X[i-1], X[i], X[i], X[i]])
            elif i==4:
                K = np.array([X[i-4], X[i-3], X[i-2], X[i-1], X[i], X[i]])
            elif i==5:
                K = np.array([X[i-5], X[i-4], X[i-3], X[i-2], X[i-1], X[i]])
            else:
                K = np.array([0.4*X[i-5], 0.5*X[i - 4], 0.6*X[i - 3], 0.7*X[i - 2], 0.8*X[i - 1], X[i]])

            S.append(K)

        return S
            #np.save( TR_BATCH_DIR+index + '__' + str(i),K)

def call(inp , out):
    IMG_SIZE = (224,224)
    Model_DIR = './model/'
    main(inp , out)
    create_training_data()