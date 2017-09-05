# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:48:38 2017

@author: Think
"""



import os
import random
import utlis
import numpy as np

import nibabel as nib

class GetData():
    def __init__(self,data_dir):
        self.source_list = []

        self.label_dir = os.path.join(data_dir, "Labels")
        self.image_dir = os.path.join(data_dir, "Images")
        self.mean = 447.140490993
        self.std = 303.824365028  #这是MM-WHS 2017 whole heart segmentation 的均值和方差
        examples = 0
        self.i = 0
        self.j = 0
        self.k = 0
        
        filelist = os.listdir(self.image_dir)
        filelist.sort()
        for file in filelist:
            if not file.endswith(".nii.gz"):
                continue
            try:
                self.source_list.append(file)
                examples = examples +1
            except Exception as e:
                print(e)
        
        print("finished loading images")
        self.examples = examples
        print("Number of examples found: ", examples)
        
        
    def next_batch(self,batch_size):
        images_list = []
        labels_list = []
        for i in range(batch_size):
            image,label = utlis.ChooseOneFile(self.image_dir,self.label_dir)
            
            
            patch_size = 64
            image_patch,label_patch = utlis.TakeOnePatch(image,label,patch_size)
            
            #convert the image_data's shape to (x,y,z,channels)
            image_x,image_y,image_z = image_patch.shape
            image_patch = image_patch[:,:,:,None]
            
            #在进行multi-label分割的时候，应该先把label变换成[size_x,size_y,size_z,n_class]的形式，把每一种label分开

            class0 = (label_patch==500).astype(np.int16) #the left ventricle blood cavity
            class1 = (label_patch==600).astype(np.int16) # the right ventricle blood cavity
            class2 = (label_patch==420).astype(np.int16) # the left atrium blood cavity 
            class3 = (label_patch==550).astype(np.int16) #the right atrium blood cavity
            class4 = (label_patch==205).astype(np.int16) # the myocardium of the left ventricle
            class5 = (label_patch==820).astype(np.int16) #the ascending aorta
            class6 = (label_patch==850).astype(np.int16) #the pulmonary artery 
            class7 = (label_patch==0).astype(np.int16) #background
            label_class = np.stack((class0,class1,class2,class3,class4,class5,class6,class7),axis=3)
            #在分割成(64,64,64)的时候，可能会出现(64,63,64)的边界情况，需要加上判断
            if image_patch.shape==(64,64,64,1):
                image_patch = (image_patch-self.mean)/self.std
                images_list.append((image_patch).astype(np.float32)) 
                labels_list.append((label_class).astype(np.int16))
            else:
                print("the shape of input image is not (64,64,64)")
                
        images = np.asarray(images_list)
        labels = np.asarray(labels_list)
        
        return images,labels
    
    def next_batch_order(self,batch_size,filename,cube_size,stride,last_point):
        images_list = []
        labels_list = []
        
        image = nib.load(os.path.join(self.image_dir,filename))
        label = nib.load(os.path.join(self.label_dir,filename))
        image_data = image.get_data()
        label_data = label.get_data()
        x,y,z = label_data.shape
    
        number = 0
        for i in range(self.i,x-cube_size,stride):
            if number == batch_size:
                break
            for j in range(self.j, y-cube_size, stride):
                if number == batch_size:
                    break
                for k in range(self.k, z-cube_size, stride):

                    image_patch = image_data[i:i+cube_size,j:j+cube_size,k:k+cube_size]
                    label_patch = label_data[i:i+cube_size,j:j+cube_size,k:k+cube_size]
                    
                    #convert the image_data's shape to (x,y,z,channels)
                    image_x,image_y,image_z = image_patch.shape
                    image_patch = image_patch[:,:,:,None]
                    
                    #在进行multi-label分割的时候，应该先把label变换成[size_x,size_y,size_z,n_class]的形式，把每一种label分开
        
                    class0 = (label_patch==500).astype(np.int16) #the left ventricle blood cavity
                    class1 = (label_patch==600).astype(np.int16) # the right ventricle blood cavity
                    class2 = (label_patch==420).astype(np.int16) # the left atrium blood cavity 
                    class3 = (label_patch==550).astype(np.int16) #the right atrium blood cavity
                    class4 = (label_patch==205).astype(np.int16) # the myocardium of the left ventricle
                    class5 = (label_patch==820).astype(np.int16) #the ascending aorta
                    class6 = (label_patch==850).astype(np.int16) #the pulmonary artery 
                    class7 = (label_patch==0).astype(np.int16) #background
                    label_class = np.stack((class0,class1,class2,class3,class4,class5,class6,class7),axis=3)
                    #在分割成(64,64,64)的时候，可能会出现(64,63,64)的边界情况，需要加上判断
                    if image_patch.shape==(64,64,64,1):
                        image_patch = (image_patch-self.mean)/self.std
                        images_list.append((image_patch).astype(np.float32)) 
                        labels_list.append((label_class).astype(np.int16))
                    else:
                        print("the shape of input image is not (64,64,64)")
                        
                    number = number + 1
                    if number == batch_size:
                        break
                    
        self.i = i
        self.j = j
        self.k = k

        images = np.asarray(images_list)
        labels = np.asarray(labels_list)
        
        return images,labels
    
    def next_batch_order_2(self,batch_size,filename,cube_size,stride,last_point):
        images_list = []
        
        image = nib.load(os.path.join(self.image_dir,filename))
      
        image_data = image.get_data()
       
        x,y,z = image_data.shape
    
        number = 0
        for i in range(self.i,x-cube_size,stride):
            if number == batch_size:
                break
            for j in range(self.j, y-cube_size, stride):
                if number == batch_size:
                    break
                for k in range(self.k, z-cube_size, stride):

                    image_patch = image_data[i:i+cube_size,j:j+cube_size,k:k+cube_size]
                    
                    #convert the image_data's shape to (x,y,z,channels)
                    image_x,image_y,image_z = image_patch.shape
                    image_patch = image_patch[:,:,:,None]

                    #在分割成(64,64,64)的时候，可能会出现(64,63,64)的边界情况，需要加上判断
                    if image_patch.shape==(64,64,64,1):
                        image_patch = (image_patch-self.mean)/self.std
                        images_list.append((image_patch).astype(np.float32)) 
                    else:
                        print("the shape of input image is not (64,64,64)")
                        
                    number = number + 1
                    if number == batch_size:
                        break
                    
        self.i = i
        self.j = j
        self.k = k

        images = np.asarray(images_list)
        
        return images
                    
       
        
        