#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Crop images according to yolo object detection coordinates
<object-class> <x_center> <y_center> <width> <height>

Only working with one object class, so we ignore <object-class>

Handles cases in which there are 
multiple lines of coordinates and no lines of coordinates

"""
import cv2
import os
import numpy as np
import math

# directory must have a txt file for each img with yolo coordinates
imgs_and_txts_path = '/Users/gcampidilli/Documents/IRV/UROP/EWM/EWM_Labeled' 
files = os.listdir(imgs_and_txts_path)
# for file list provided, remove .txt and .jpg extensions
for f in range(len(files)):
        files[f] = files[f][:-3]
    # remove duplicates
files = np.unique(files)

# location for cropped images
output_imgs = '/Users/gcampidilli/Documents/IRV/video_splicer/extracted_images/cropped_imgs'
os.makedirs(output_imgs, exist_ok = True) 


def img_crop(files, imgs_and_txts_path, output_imgs):
    count = 0
    for file in files:    
        print(file)
        # open text file
        f = open(os.path.join(imgs_and_txts_path,file+'txt'),'r')
        lines = f.readlines()
        
        if len(lines) == 0:
            next
        else:
            # load image
            if os.path.exists(os.path.join(imgs_and_txts_path,file+'jpg')):
                img = cv2.imread(os.path.join(imgs_and_txts_path,file+'jpg'))
            elif os.path.exists(os.path.join(imgs_and_txts_path,file+'png')):
                img = cv2.imread(os.path.join(imgs_and_txts_path,file+'png'))
            else:
                next
            h,w,_ = img.shape
            print(h,w)
            # iterate through lines in txt file
            for line in range(len(lines)):
                print(line)
                # transform string of coordinates into an array        
                coords_str = lines[line].split()[1:]
                # convert string array to integers
                coords_norm = [float(i) for i in coords_str]
                # crop image with txt file coordinates
                x,y,wx,hy = int(coords_norm[0]*w),int(coords_norm[1]*h),int(coords_norm[2]*w),int(coords_norm[3]*h)
                # make sure indicies are within bounds of image
                xleft, xright, ybottom, ytop = max(math.floor(y-hy/2)+5,0), min(math.floor(y+hy/2)+5,h), max(math.floor(x-wx/2)+5,0), min(math.floor(x+wx/2)+5,w)
                print(xleft, xright, ybottom, ytop)
                crop_img = img[xleft:xright, ybottom:ytop]            
                # save image
                cv2.imwrite(os.path.join(output_imgs,'cropped_'+str(line)+'_'+file+'jpg'),crop_img)
        print(count)
        count = count + 1

img_crop(files, imgs_and_txts_path, output_imgs)








