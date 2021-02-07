###Import Files

import os
import SimpleITK as sitk
import glob



###Declare Variables

files_loc    = "Data_Online/NITRC/Ready/"



###Read in Folders

folders = os.listdir(files_loc)
print(folders)

for folder in folders:
    
    file_path     = files_loc + folder + "/Flair_resized/*.png"
    map_path      = files_loc + folder + "/map_resized/*.png"
   
    file_save     = files_loc + folder + "/Flair_resized.nii"
    map_save      = files_loc + folder + "/map_resized.nii"
    
    file_names = glob.glob(file_path)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    vol = reader.Execute()
    sitk.WriteImage(vol, file_save)
    
    file_names = glob.glob(map_path)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    vol = reader.Execute()
    sitk.WriteImage(vol, map_save)