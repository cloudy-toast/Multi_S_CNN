"""
Created on Mon Oct 17 20:09:14 2016
@author: Weidi XIE
Description:  Training function 
"""
import numpy as np
from Read_US_aug import get_data
import pdb
import os
from build_multi_model import get_unet
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

data_base_path = 'D:\\Ruobing\\Mat_data\\Scans_3d\\Train'
label_base_path = 'D:\\Ruobing\\Mat_data\\Mask_3D\\Train'
imgList = os.listdir('D:\\Ruobing\\Mat_data\\Mask_3D\\Train\\CC')
batch_sz = 120#20
sub_batch_sz = 10
epoch = 50

def step_decay(epoch):
    initial_lrate = 1e-3
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
    return np.float(lrate)
 
def train_():
    print('-'*30)
    print('Creating and compiling the 3D FCN Model.')
    print('-'*30)    
    # Need to calculate the mean of all the volumes, save it.
    mean = 22.6073
    model = get_unet()
    model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)
    print('...Fitting model...')
    print('-'*30)
    fileList = imgList
    
    for i in range(150):
        imgs,imgs_mask1,imgs_mask2,imgs_mask3,imgs_mask4,imgs_mask5 = get_data(data_base_path,label_base_path,fileList,batch_sz)
        imgs -= mean
        change_lr = LearningRateScheduler(step_decay)
       # if i > 0:
        model.load_weights('unet.hdf5')
            
        model.fit( {'input' : imgs},
                  {'task1_output' : imgs_mask1,
                   'task2_output' : imgs_mask2,
                   'task3_output' : imgs_mask3,
                   'task4_output' : imgs_mask4,
                   'task5_output' : imgs_mask5},
                   batch_size = sub_batch_sz,
                   nb_epoch =20,
                   verbose=1, shuffle=True,
                   callbacks=[model_checkpoint,change_lr])
        
if __name__ == '__main__':
    train_()