### Import Files

import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imsave
import random



###Import Models

models = []
models.append('Data/OHSU/model/U-Net_OHSU_Only')
models.append('Data/OHSU_LIT/model/U-Net_OHSU_LIT_Train')
models.append('Data/OHSU_NITRC/model/U-Net_OHSU_NITRC_train')
models.append('Data/OHSU_LIT_NITRC/model/U-Net_OHSU_LIT_NITRC_Train')

###Declare Variables

files_loc  = 'Data/OHSU/test/'
height     = 320
width      = 240
depth      = 320
model_load = models[3]
save_loc   = 'Results/'
threshold_constant = 0.2



###List Folders

files      = os.listdir(files_loc)
file_count = len(files)
print(file_count, "Folders Discovered")



### Create Random List

test_sel = []
for i in range(0, 12):
    test_sel.append(random.randint(0, (file_count*depth)))



###Creat Arrays

X_initial = np.zeros((depth*file_count, width, height),dtype=np.uint16)
Y_initial = np.zeros((depth*file_count, width, height),dtype=np.bool)
X_test    = np.zeros((len(test_sel), width, height),dtype=np.uint16)
Y_test    = np.zeros((len(test_sel), width, height),dtype=np.bool)
Y_true    = np.zeros((len(test_sel), width, height),dtype=np.bool)
Y_pred    = np.zeros((len(test_sel), width, height),dtype=np.bool)

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

print("Removing Redundant Files")
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

for x, num in enumerate(test_sel):
    X_test[x] = X_small[num]
    Y_test[x] = Y_small[num]
    
print("Initial Arrays of the Following Size Created")        
print(X_test.shape)
print(Y_test.shape)

X_test = np.reshape(X_test,(len(X_test),width,height,1))
Y_test = np.reshape(Y_test,(len(Y_test),width,height,1))

print("Arrays Reshaped to")
print(X_test.shape) 
print(Y_test.shape)



#Cleanup

del X_initial, Y_initial, X_small, Y_small



#Load Model

print("Loading Model: ", model_load)
model = tf.keras.models.load_model(model_load)
model.summary()



#Predict Model

print("Making Predictions...")

results = model.predict(X_test,batch_size=16)


#Show Predictions

print("Reshaping Predictions...")

results = np.reshape(results,(len(results),width,height))
X_test = np.reshape(X_test,(len(X_test),width,height))
Y_test = np.reshape(Y_test,(len(Y_test),width,height))





#Threshold Images
for num, x in enumerate(X_test):
    Y_true[num] = Y_test[num]
    result = results[num]
    result_max = np.amax(result)
    threshold = result_max * threshold_constant
    
    for x, data in enumerate(result):
        for y , data2 in enumerate(data):
            if result[x][y] > threshold:
                result[x][y] = 1
            if result[x][y] < threshold:
                result[x][y] = 0
    
    Y_pred[num] = result
    

      
#Calculate Metrics
total_true = 0
pred_true  = 0

for num, data in enumerate(Y_true):
    for x, data2 in enumerate(data):
        for y, data3 in enumerate(data2):
            if Y_true[num][x][y] == True:
                total_true = total_true + 1
                if Y_pred[num][x][y] == True:
                    pred_true = pred_true + 1

spec_total = 0
spec_hit   = 0
for num, data in enumerate(Y_pred):
    for x, data2 in enumerate(data):
        for y, data3 in enumerate(data2):
            if Y_pred[num][x][y] == True:
                spec_total = spec_total + 1
                if Y_true[num][x][y] == True:
                    spec_hit = spec_hit + 1
                    
print(model_load)
print("Sen:", pred_true/total_true)
print("Spec:", spec_hit/spec_total)


for x, data in enumerate(X_test):
    
    plt.imshow(X_test[x], cmap = 'gray')
    plt.imshow(Y_true[x], cmap='pink', alpha=0.5)
    plt.imshow(Y_pred[x], cmap='jet', alpha=0.5)
    plt.title(x)
    plt.show()
    
    X_test_path = save_loc + 'X_'+ str(x) + '.png'
    Y_test_path = save_loc + 'Y_'+ str(x) + '.png'
    pred_path   = save_loc + 'P_'+ str(x) + '.png'
    
    imsave(X_test_path, X_test[x])  
    Image.fromarray(Y_true[x]).save(Y_test_path)
    Image.fromarray(Y_pred[x]).save(pred_path)    