# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:42:48 2017

@author: Think
"""

import nibabel as nib
import os
import numpy as np
from scipy import ndimage


from fFindImageBoundaryCoordinate3D import fFindImageBoundaryCoordinate3D
from data_augmentation import augmentation

def CreatNii_save(data,save_dir,filename,affine):
    """
    data: the original data
    save_dir: the directory to save the nii
    filename : the name of this nii
    affine: affine matrix
    """

    img = nib.Nifti1Image(data,affine)  #新建的图片与原始的affine不能变
    img.header.get_xyzt_units()
    
    nib.save(img,os.path.join(save_dir,filename))
    

def CutBoundingBox(load_dir,save_dir,filename):
    """
    load_dir: is the directory consists of the nii files
    save_dir : is the directory which saves the boundingbox nii
    filename : the nii should be loaded
    """
    image = nib.load(os.path.join(load_dir,filename))
    xdim, ydim, zdim = fFindImageBoundaryCoordinate3D(image.get_data(),15)
    
    xdim = xdim.astype(int)
    ydim = ydim.astype(int)
    zdim = zdim.astype(int) #convert float to int
    
    image_data = image.get_data()
    image_bouding_box = image_data[xdim[0]:xdim[1],ydim[0]:ydim[1],zdim[0]:zdim[1]]
    bounding_box_filename = filename.strip(".nii.gz")+"_boundingBox.nii.gz"
    
    CreatNii_save(image_bouding_box,save_dir,bounding_box_filename,image.affine)
    
def TakeOnePatch(image,label,patch_size):
    """
    image,label: take patch from these image and label
    patch_size: a int number, patch size is (patch_size,patch_size,patch_size) by default
    return: corresponding image_patch and label_patch 
    """
    
    image_data = image.get_data()
    label_data = label.get_data()
    
    x,y,z = label_data.shape
    
    i = np.random.randint(0,x-patch_size)
    j = np.random.randint(0,y-patch_size)
    k = np.random.randint(0,z-patch_size)
    
    image_cube_data = image_data[i:i+patch_size,j:j+patch_size,k:k+patch_size]
    label_cube_data = label_data[i:i+patch_size,j:j+patch_size,k:k+patch_size]
    
    image_patch,label_patch = augmentation(image_cube_data,label_cube_data)
    
    return image_patch,label_patch

def ChooseOneFile(image_dir,label_dir):
    """
    choose one nii randomly from 20 cases
    corresponding image and label have the same name in different directories
    """
    
    directory = os.listdir(image_dir)
    
    i = np.random.randint(0,len(directory))
    
    file = directory[i]
    
    image = nib.load(os.path.join(image_dir,file))
    label = nib.load(os.path.join(label_dir,file))
    
    return image,label


#remove small islands from binary volume
def CC(Map):
    label_img, cc_num = ndimage.label(Map)
    cc_areas = ndimage.sum(Map, label_img, range(cc_num+1))
    area_mask = (cc_areas < np.max(cc_areas))
    label_img[area_mask[label_img]] = 0
    return (label_img!=0).astype(np.int16)    
    