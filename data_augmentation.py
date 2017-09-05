# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:59:10 2017

@author: Think
"""

from skimage import transform
import math
import numpy as np

def augmentation(src_image,src_label):
    theta = math.pi/2

    
    i = np.random.randint(0,3)
    j = np.random.randint(0,3)
    #Rotation matrix, angle theta, translation tx,ty,tz
    tx=0
    ty=0
    tz=0
    H = np.array([[math.cos(i*theta), math.sin(i*theta), 0, tx],
                           [-math.sin(i*theta), math.cos(i*theta), 0 ,ty],
                           [0, 0, 1, tz],
                           [0, 0, 0, 1]])
    
    
    #Translation matrix to shift the image center to the origin
    shift_y, shift_x = np.array(src_image.shape[:2]) / 2.
    tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(30))
    tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
    
    dst_image = transform.warp(src_image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
    dst_label = transform.warp(src_image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
    
#    offset = 0
#    dst_image = transform.homography(src_image,np.linalg.inv(T).dot(H).dot(T),offset=offset,mode='nearest')
#    dst_label = transform.homography(src_label,np.linalg.inv(T).dot(H).dot(T),offset=offset,mode='nearest')
    
    image = np.flip(dst_image,axis=j)
    label = np.flip(dst_label,axis=j)
    
    return image,label