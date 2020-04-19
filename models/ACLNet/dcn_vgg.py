'''
This code is part of the Keras VGG-16 model
'''
from __future__ import print_function
from __future__ import absolute_import

from keras.models import Sequential
from keras.layers import MaxPooling2D, Conv2D
from keras.utils.data_utils import get_file

TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def dcn_vgg():
    model = Sequential()
    # conv_1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(224, 224, 3)))#batch__input_shape=(224,224,3)
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # conv_2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # conv_3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', padding='same'))

    # conv_4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    #
    model.add(MaxPooling2D((2, 2), strides=(1, 1), name='block4_pool', padding='same'))

    # conv_5
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', dilation_rate=(2, 2)))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', dilation_rate=(2, 2)))
    # model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', dilation_rate=(2, 2)))
    #
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    # Load weights
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP,
                            cache_subdir='models')
    model.load_weights(weights_path)

    return model
