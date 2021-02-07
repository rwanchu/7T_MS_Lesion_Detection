### Import Files

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from sklearn.utils import shuffle
import datetime



###Declare Variables

files_loc  = 'Data/OHSU/train/'
height     = 320
width      = 240
depth      = 320
model_path = 'OHSU_UNET.model'
model_out  = 'Data/OHSU/model/U-Net_OHSU_Only'
log_dir = os.path.join("Data","OHSU","model",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),)



###List Folders

files      = os.listdir(files_loc)
file_count = len(files)
print(file_count, "Folders Discovered")



###Creat Arrays

X_initial = np.zeros((depth*file_count, width, height),dtype=np.uint16)
Y_initial = np.zeros((depth*file_count, width, height),dtype=np.bool)

x = 0
y = 0



#Scan Files in

for file in files:
             
    flair_path = files_loc + file + '/flair.bse.nii'
    map_path   = files_loc + file + '/mask.nii'
    print(flair_path)
    
    
    if os.path.exists(flair_path):
        print("Flair Volume Found")
        scan = nib.load(flair_path)
        scan_data = scan.get_fdata()
        num = 0
        while num < depth-1:
            X_initial[x] = scan_data[:,:,num]
            x = x + 1
            num = num + 1
        
    if os.path.exists(map_path): 
        print("Mask Volume Found")
        scan = nib.load(map_path)
        scan_data = scan.get_fdata()
        num = 0
        while num < depth-1:
            Y_file = scan_data[:,:,num]
            Y_file = np.flip(Y_file,0)
            Y_initial[y] = Y_file
            y = y + 1
            num = num + 1
        
print("Files Scanned In")



#Remove Images with no values

print("Removing Redundant Files...")
X_small = []
Y_small = []
x = 0
y = 0

for count, data in enumerate(X_initial):
     
    if X_initial[count].max() == 0:
       pass
    else:
        X_small.append(X_initial[count])
        Y_small.append(Y_initial[count])



#Create Final Array 

X_train = np.asarray(X_small, dtype=np.uint16)
Y_train = np.asarray(Y_small, dtype=np.uint16)

print("Initial Arrays of the Following Size Created")        
print(X_train.shape)
print(Y_train.shape)

X_train = np.reshape(X_train,(len(X_train),width,height,1))
Y_train = np.reshape(Y_train,(len(Y_train),width,height,1))

print("Arrays Reshaped to")
print(X_train.shape) 
print(Y_train.shape)



#Cleanup

del X_initial, Y_initial, X_small, Y_small



#Shuffle Dataset

print("Shuffling Dataset...")
X_train, Y_train = shuffle(X_train, Y_train)



#Load Model

print("Loading Model...")
model = tf.keras.models.load_model(model_path)
model.summary()



#Train Model

print("Training Starting...")

tboard_cb = tf.keras.callbacks.TensorBoard(log_dir)
save_cb   = tf.keras.callbacks.ModelCheckpoint(filepath=model_out, verbose=1)
early_cb  = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=1)
cb_list   = [tboard_cb, save_cb, early_cb] 

results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=10, callbacks=cb_list)

print("Training Complete. Model saved at", model_out)