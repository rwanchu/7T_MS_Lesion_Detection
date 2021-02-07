###Import Files

import cv2
import numpy as np
import os



###Declare Variables

size_x       = 200
height       = 320
width        = 240
files_loc    = "Data/LIT/train/"



###Read in Folders

folders = os.listdir(files_loc)
print(folders)

for folder in folders:
    
    file_path     = files_loc + folder + "/flair/"
    save_path     = files_loc + folder + "/flair_resized/"
    
    map_path      = files_loc + folder + "/mask/"
    map_save_path = files_loc + folder + "/mask_resized/"
    
    os.mkdir(save_path)
    os.mkdir(map_save_path)
   
    files = os.listdir(file_path)
    maps  = os.listdir(map_path)
    
    for num, data in enumerate(files):
        
        img_loc = file_path + data
        img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
        
        shape = img.shape
    
        total_x = shape[1]
        x_center = total_x/2
        x_min = int(x_center - size_x/2)
        x_max = int(x_center + size_x/2)
        
        total_y = shape[0]
        y_min = 40
        y_max = 295
        
        crop_img = img[y_min:y_max, x_min:x_max]
        
        im = np.array(crop_img)
        im = 255.0 * (im / 255.0)**(1.6 / 1)
        
        im = cv2.resize(im,(width, height))
           
        save_loc = save_path + data
        cv2.imwrite(save_loc,im)
        
    
    for num, data in enumerate(maps):
        
        img_loc = map_path + data
        img = cv2.imread(img_loc, cv2.IMREAD_GRAYSCALE)
        
        shape = img.shape
    
        total_x = shape[1]
        x_center = total_x/2
        x_min = int(x_center - size_x/2)
        x_max = int(x_center + size_x/2)
        
        total_y = shape[0]
        y_min = 40
        y_max = 295
        
        crop_img = img[y_min:y_max, x_min:x_max]
        
        im = np.array(crop_img)
        im = 255.0 * (im / 255.0)**(1.6 / 1)
           
        im = cv2.resize(im,(width, height)) 
        
        save_loc = map_save_path + data
        cv2.imwrite(save_loc,im)