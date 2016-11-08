"""
Created on Mon Oct 17 17:24:42 2016
@author: Ruobing Huang
"""

from __future__ import print_function
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers import Dropout, Activation
from keras.optimizers import SGD, RMSprop
import pdb

image_length =128
image_width  =128
image_height =128

def get_unet():
    inputs = Input((image_length, image_width, image_height), name = 'input')
    conv1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering = 'tf',init="orthogonal")(inputs)
    conv1 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',dim_ordering = 'tf',init="orthogonal")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering = 'tf',init="orthogonal")(pool1)
    conv2 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',dim_ordering = 'tf',init="orthogonal")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering = 'tf',init="orthogonal")(pool2)
    conv3 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering = 'tf',init="orthogonal")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
   
    # ==========================================================================
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering = 'tf',init="orthogonal")(pool3)
    drop4 = Dropout(0.5)(conv4)
    conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same',dim_ordering = 'tf',init="orthogonal")(drop4)
    conv4 = Dropout(0.5)(conv4)
    # ==========================================================================
    #                                  Task 1
    # ==========================================================================
    t1_conv5 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(conv4)
    t1_act5 = merge([UpSampling2D(size=(2, 2))(t1_conv5), conv3], mode='concat', concat_axis= -1)
    
    t1_conv6 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(t1_act5)
    t1_act6 =  merge([UpSampling2D(size=(2, 2))(t1_conv6), conv2], mode='concat', concat_axis= -1)
    
    t1_pred =  Convolution2D(1, 1, 1, activation='sigmoid',dim_ordering = 'tf',init='orthogonal', name='task1_output')(t1_act6)
    # ==========================================================================
    #                                  Task 2
    # ==========================================================================
    t2_conv5 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(conv4)
    t2_act5 = merge([UpSampling2D(size=(2, 2))(t2_conv5), conv3], mode='concat', concat_axis= -1)
    
    t2_conv6 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(t2_act5)
    t2_act6 =  merge([UpSampling2D(size=(2, 2))(t2_conv6), conv2], mode='concat', concat_axis= -1)
    
    t2_pred =  Convolution2D(1, 1, 1, activation='sigmoid',dim_ordering = 'tf',init='orthogonal', name='task2_output')(t2_act6)
    # ==========================================================================
    #                                  Task 3
    # ==========================================================================
    t3_conv5 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(conv4)
    t3_act5 = merge([UpSampling2D(size=(2, 2))(t3_conv5), conv3], mode='concat', concat_axis= -1)
    
    t3_conv6 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(t3_act5)
    t3_act6 =  merge([UpSampling2D(size=(2, 2))(t3_conv6), conv2], mode='concat', concat_axis= -1)

    t3_pred =  Convolution2D(1, 1, 1, activation='sigmoid',dim_ordering = 'tf',init='orthogonal', name='task3_output')(t3_act6)
    # ==========================================================================
    #                                  Task 4
    # ==========================================================================
    t4_conv5 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(conv4)
    t4_act5 = merge([UpSampling2D(size=(2, 2))(t4_conv5), conv3], mode='concat', concat_axis= -1)
    
    t4_conv6 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(t4_act5)
    t4_act6 =  merge([UpSampling2D(size=(2, 2))(t4_conv6), conv2], mode='concat', concat_axis= -1)
    
    t4_pred =  Convolution2D(1, 1, 1, activation='sigmoid',dim_ordering = 'tf',init='orthogonal', name='task4_output')(t4_act6)
   
    # ==========================================================================
    #                                  Task 5
    # ==========================================================================
    t5_conv5 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(conv4)
    t5_act5 = merge([UpSampling2D(size=(2, 2))(t5_conv5), conv3], mode='concat', concat_axis= -1)
    
    t5_conv6 = Convolution2D(256, 3, 3, activation = 'relu', init = 'orthogonal', dim_ordering = 'tf',border_mode='same')(t5_act5)
    t5_act6 =  merge([UpSampling2D(size=(2, 2))(t5_conv6), conv2], mode='concat', concat_axis= -1)
    
    t5_pred =  Convolution2D(1, 1, 1, activation='sigmoid',dim_ordering = 'tf',init='orthogonal',name='task5_output')(t5_act6)
    # ==========================================================================
    model = Model(input = inputs, output= [t1_pred, t2_pred, t3_pred, t4_pred, t5_pred])
    # ==========================================================================
    opt = RMSprop(lr = 1e-3)
    model.compile(optimizer = opt,
                  loss={'task1_output': 'binary_crossentropy',
                        'task2_output': 'binary_crossentropy',
                        'task3_output': 'binary_crossentropy',
                        'task4_output': 'binary_crossentropy',
                        'task5_output': 'binary_crossentropy'},
                  metrics={'task1_output': 'acc',
                           'task2_output': 'acc',
                           'task3_output': 'acc',
                           'task4_output': 'acc',
                           'task5_output': 'acc'},
                  loss_weights = {'task1_output': 1,
                                  'task2_output': 1,
                                  'task3_output': 1,
                                  'task4_output': 1,
                                  'task5_output': 1})
    return model