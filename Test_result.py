# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 11:43:19 2016

@author: spet3779
"""

import numpy as np
from Read_US_aug import get_data
from Read_US_aug import random_transform
import os
import scipy
import matplotlib.pyplot as plt
from scipy.io import loadmat


data_path = 'D:\\Ruobing\\Mat_data\\Scans_3d\\Validation'
label_path = 'D:\\Ruobing\\Mat_data\\Mask_3D\\Validation'
img_list = os.listdir('D:\\Ruobing\\Mat_data\\Mask_3D\\Validation\\CC')
batch_sz = 2
sub_batch_sz = 2
imgs,imgs_mask1,imgs_mask2,imgs_mask3,imgs_mask4,imgs_mask5 =  get_data(data_path, label_path, img_list, batch_sz, view = 'Axial', train = True, rot_rng=20, flip_pb=0.5, scale_rng=0.75)
x=imgs[0,:,:,40]
plt.imshow(x)    
x1=imgs_mask2[0,:,:,0]
plt.imshow(x1)


#plt.show(x)
baseName = img_list[0][:-3]
filename =  os.path.join(data_path, baseName + 'mat')

label_path1 = os.path.join(label_path, 'CC',  baseName + 'mat')
label_path2 = os.path.join(label_path, 'CE',  baseName + 'mat')
label_path3 = os.path.join(label_path,'CM', baseName + 'mat')
label_path4= os.path.join(label_path,'LV', baseName + 'mat')
label_path5 = os.path.join(label_path,'Tha',baseName + 'mat')

dict=loadmat(filename)
img = dict['img']


class Option:
    def __init__(self,rotation_range,flip_prob,scale_range):
       # self.name= 0
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        
opt = Option(30,0.5,0.5)    

dict=loadmat(label_path1)
label1=dict['anno']

dict=loadmat(label_path2)
label2=dict['anno']

dict=loadmat(label_path3)
label3=dict['anno']

dict=loadmat(label_path4)
label4=dict['anno']

dict=loadmat(label_path5)
label5=dict['anno']
 

def random_transform (opt,img,label1,label2,label3,label4,label5):
    #Rotate random angle within view based on opt.rotation_range
    if opt.rotation_range:
        theta = np.random.uniform(-opt.rotation_range, opt.rotation_range)
    else:
        theta = 0
    img = rotate(img,theta,axes=(0, 1),reshape=False)
    label1 = rotate(label1,theta,axes=(0, 1),reshape=False)
    label2 = rotate(label2,theta,axes=(0, 1),reshape=False)
    label3 = rotate(label3,theta,axes=(0, 1),reshape=False)
    label4 = rotate(label4,theta,axes=(0, 1),reshape=False)
    label5 = rotate(label5,theta,axes=(0, 1),reshape=False)
   
    #flip img based on flip_prob
    random_no = random.random()
    if opt.flip_prob and opt.flip_prob > random_no:
        img = np.flipud(img)
        label1 = np.flipud(label1)
        label2 = np.flipud(label2)
        label3 = np.flipud(label3)
        label4 = np.flipud(label4)
        label5 = np.flipud(label5)
        
    #crop image based on scale range 
    if opt.scale_range:
        scalex = (1-opt.scale_range) * random.random()+ opt.scale_range
        scaley = (1-opt.scale_range) * random.random()+ opt.scale_range

    x1 = random.randint(0,int((1-scalex)*img.shape[0]))
    x2 = x1+int(scalex*img.shape[0])
    y1 = random.randint(0,int((1-scaley)*img.shape[1]))
    y2 = y1+int(scaley*img.shape[1])

    img_transformed = img[x1:x2,y1:y2,:]    
    label1= label1[x1:x2,y1:y2,:]
    label2 = label2[x1:x2,y1:y2,:]
    label3 = label3[x1:x2,y1:y2,:]
    label4 = label4[x1:x2,y1:y2,:]
    label5 = label5[x1:x2,y1:y2,:]

    #crop black area
    Points = np.nonzero(img_transformed)
    x1=min(Points[0])
    y1=min(Points[1])
    z1=min(Points[2])
    x2=max(Points[0])
    y2=max(Points[1])
    z2=max(Points[2])
    img_transformed = img[x1:x2,y1:y2,z1:z2]    
    label1 = label1[x1:x2,y1:y2,z1:z2]
    label2 = label2[x1:x2,y1:y2,z1:z2]
    label3 = label3[x1:x2,y1:y2,z1:z2]
    label4 = label4[x1:x2,y1:y2,z1:z2]
    label5 = label5[x1:x2,y1:y2,z1:z2]

    label_im1=np.sum(label1,axis=2)>0
    label_im2=np.sum(label2,axis=2)>0
    label_im3=np.sum(label3,axis=2)>0
    label_im4=np.sum(label4,axis=2)>0
    label_im5=np.sum(label5,axis=2)>0

    return img_transformed,label_im1,label_im2,label_im3,label_im4,label_im5  

img_transformed,label_im1,label_im2,label_im3,label_im4,label_im5 =random_transform (opt,img,label1,label2,label3,label4,label5)

x=img_transformed[:,:,1]
plt.imshow(x)
plt.imshow(label_im5)

for layer in model.layers:
    h=layer.get_weights()
    print(h)