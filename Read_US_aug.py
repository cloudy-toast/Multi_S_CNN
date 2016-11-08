"""
Created on Wed Oct 19 15:55:17 2016
# Author : Ruobing Huang 
# Input volume is 128 x 128 x 128
# Prediction mask will be 64 x 64 x 1
# this get_data function load images in the format length x width x height
# it is tensorflow style
#Option creates attributes to randomly transform input images
#i.e.opt = Option(10,0.7,0.8)
"""

from scipy.io import loadmat
import numpy as np
import os
import scipy.ndimage
import pdb
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
import random

image_length = 128.0
image_width  = 128.0
image_height = 128.0

label_width = 64.0
label_height = 64.0

class Option:
    def __init__(self,rotation_range,flip_prob,scale_range):
       # self.name= 0
        self.rotation_range = rotation_range
        self.flip_prob = flip_prob
        self.scale_range = scale_range

# img_list needs to be changed every time, when calling get_data
def get_data(data_path, label_path, img_list, total, view = 'Axial', train = True, rot_rng=30, flip_pb=0.5, scale_rng=0.6):
    imgs = np.ndarray((total,image_length,image_width,image_height),dtype=np.float32)
    imgs_mask1=np.ndarray((total,image_length/2,image_width/2,1),dtype=np.uint8)
    imgs_mask2=np.ndarray((total,image_length/2,image_width/2,1),dtype=np.uint8)
    imgs_mask3=np.ndarray((total,image_length/2,image_width/2,1),dtype=np.uint8)
    imgs_mask4=np.ndarray((total,image_length/2,image_width/2,1),dtype=np.uint8)
    imgs_mask5=np.ndarray((total,image_length/2,image_width/2,1),dtype=np.uint8)
    #Set the option to passed in value
    opt = Option(rot_rng,flip_pb,scale_rng)
    
    for i in range(total):
        baseName = img_list[i][:-3]
        filename =  os.path.join(data_path, baseName + 'mat')
        
        label_path1 = os.path.join(label_path, 'CC',  baseName + 'mat')
        label_path2 = os.path.join(label_path, 'CE',  baseName + 'mat')
        label_path3 = os.path.join(label_path,'CM', baseName + 'mat')
        label_path4= os.path.join(label_path,'LV', baseName + 'mat')
        label_path5 = os.path.join(label_path,'Tha',baseName + 'mat')
        
        dict=loadmat(filename)
        img_res = dict['img']
        #Rotate the img according to the view
        img_res = flip_axis2view(img_res, view)
        
        if train == True:
            dict=loadmat(label_path1)
            mask1=dict['anno']
            mask1=flip_axis2view(mask1, view)
            dict=loadmat(label_path2)
            mask2=dict['anno']
            mask2=flip_axis2view(mask2, view)
            dict=loadmat(label_path3)
            mask3=dict['anno']
            mask3=flip_axis2view(mask3, view)
            dict=loadmat(label_path4)
            mask4=dict['anno']
            mask4=flip_axis2view(mask4, view)
            dict=loadmat(label_path5)
            mask5=dict['anno']
            mask5=flip_axis2view(mask5, view)
            #Perform same transformation to the image and labels
            #Return trnasformed img and corresponding 2D label images
            img_res,mask1_res,mask2_res,mask3_res,mask4_res,mask5_res = random_transform (opt,img_res,mask1,mask2,mask3,mask4,mask5)
            #plt.show(mask1_res)
            
            mask1_res=preprocess_mask(mask1_res,1)
            mask2_res=preprocess_mask(mask2_res,1)
            mask3_res=preprocess_mask(mask3_res,1)
            mask4_res=preprocess_mask(mask4_res,1)
            mask5_res=preprocess_mask(mask5_res,1)
            
            #plt.show(mask1_res)
            
            imgs_mask1[i,:,:,0]= np.array([mask1_res > 0], dtype=np.int32)
            imgs_mask2[i,:,:,0]= np.array([mask2_res > 0], dtype=np.int32)
            imgs_mask3[i,:,:,0]= np.array([mask3_res > 0], dtype=np.int32)
            imgs_mask4[i,:,:,0]= np.array([mask4_res > 0], dtype=np.int32)
            imgs_mask5[i,:,:,0]= np.array([mask5_res > 0], dtype=np.int32)
        #Normalize the img to 128X128, label to 64X64
        img_res= preprocess_img(img_res)
        imgs[i,:,:,:]=np.array(img_res,dtype=np.float32)
        if i % 5 == 0:
            print('Done: {0}/{1} images'.format(i, total))
    return imgs,imgs_mask1,imgs_mask2,imgs_mask3,imgs_mask4,imgs_mask5
        
def preprocess_img(data) :       
    img_p = scipy.ndimage.zoom(data, (image_height / data.shape[0],image_length/ data.shape[1],image_width/ data.shape[2]))
    return img_p

def preprocess_mask(imgs,s)   :
    imgs_p2 = np.ndarray((image_length / (2*s), image_width / (2*s)), dtype=np.uint8)
    imgs_p2= scipy.misc.imresize(imgs, (imgs_p2.shape[0], imgs_p2.shape[1]))
    return imgs_p2
    
        
def flip_axis2view(x, view):
    if view == 'Sagittal':
        x = np.swapaxes(x,0,2)
        x = np.flipud(x)
    elif view == 'Coronal':
        x = np.swapaxes(x,1,2)
        x = np.rot90(x)
    return x
    
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