# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 20:09:14 2016
@author: Weidi XIE
@Description: Training function 
"""

import numpy as np
from Read_US_aug import get_data
import pdb
import os
import scipy
import matplotlib.pyplot as plt
from build_multi_model import get_unet

data_base_path = 'D:\\Ruobing\\Mat_data\\Scans_3d\\Validation'
label_base_path = 'D:\\Ruobing\\Mat_data\\Mask_3D\\Validation'
imgList = os.listdir('D:\\Ruobing\\Mat_data\\Mask_3D\\Validation\\CC')
batch_sz = 29
sub_batch_sz =4

def output_image_and_mask(save_path, img_mask):
    scipy.misc.imsave(save_path , 255*img_mask)#
 
def test_():
    print('-'*30)
    print('Creating and compiling the 3D FCN Model.')
    print('-'*30)    
    # need to load the mean.
    mean = 22.6073
    model = get_unet()
    model.load_weights('unet.hdf5')
    print('...Testing model...')
    print('-'*30)
    
    fileList = imgList
    imgs,imgs_mask1,imgs_mask2,imgs_mask3,imgs_mask4,imgs_mask5 = get_data(data_base_path,label_base_path,fileList,batch_sz,train = True)
    imgs -= mean
    pred = model.predict( {'input' : imgs},
                           batch_size = sub_batch_sz,
                           verbose=1,)
#    x=imgs[0,:,:,40]
#    plt.imshow(x)
#    x1=imgs_mask2[0,:,:,0]
#    plt.imshow(x1)
                         
    img_pred1 = np.array(pred[0], dtype = np.float32)
    img_pred2 = np.array(pred[1], dtype = np.float32)
    img_pred3 = np.array(pred[2], dtype = np.float32)
    img_pred4 = np.array(pred[3], dtype = np.float32)
    img_pred5 = np.array(pred[4], dtype = np.float32)
#    x3=img_pred3[0,:,:,0]
#    plt.imshow(x3*255)
#    x1=imgs_mask2[0,:,:,0]
#    plt.imshow(x1)
    
    savepath1 = 'D:\\Ruobing\\Mat_data\\Mask_3D\\predict\\CC\\'
    savepath2 = 'D:\\Ruobing\\Mat_data\\Mask_3D\\predict\\CE\\'
    savepath3 = 'D:\\Ruobing\\Mat_data\\Mask_3D\\predict\\CM\\'
    savepath4 = 'D:\\Ruobing\\Mat_data\\Mask_3D\\predict\\LV\\'
    savepath5 = 'D:\\Ruobing\\Mat_data\\Mask_3D\\predict\\Tha\\'
    
    for i in range(batch_sz):
        name = imgList[i]
        name = name[:-3] + 'png'
        sp1 = savepath1 + name
        sp2 = savepath2 + name
        sp3 = savepath3 + name
        sp4 = savepath4 + name
        sp5 = savepath5 + name
        output_image_and_mask(sp1, img_pred1[i,:,:,0])
        output_image_and_mask(sp2, img_pred2[i,:,:,0])
        output_image_and_mask(sp3, img_pred3[i,:,:,0])
        output_image_and_mask(sp4, img_pred4[i,:,:,0])
        output_image_and_mask(sp5, img_pred5[i,:,:,0])
        name = name[:-4] + 'anno.png'
        sp1 = savepath1 + name
        sp2 = savepath2 + name 
        sp3 = savepath3 + name 
        sp4 = savepath4 + name
        sp5 = savepath5 + name
        output_image_and_mask(sp1, imgs_mask1[i,:,:,0])
        output_image_and_mask(sp2, imgs_mask2[i,:,:,0])
        output_image_and_mask(sp3, imgs_mask3[i,:,:,0])
        output_image_and_mask(sp4, imgs_mask4[i,:,:,0])
        output_image_and_mask(sp5, imgs_mask5[i,:,:,0])
if __name__ == '__main__':
    test_()