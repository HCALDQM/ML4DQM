import tensorflow as tf
import utils_for_train as u
import h5py
import os
import numpy as np

from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics

import keras 
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D,Activation,BatchNormalization,LeakyReLU,UpSampling2D
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


data_folder = '../../data'
file_name = 'HCAL_digi+rechit_occ.hdf5'
group = 'DigiTask_Occupancy_depth_depth1'

input_file=h5py.File(data_folder+"/"+file_name,'r+')

data_sample= np.array(input_file[group])


print data_sample.shape

data_sample=data_sample[:,:,26:58]

print data_sample.shape




input_images=data_sample[:]
hotregion_image=[]
deadregion_image=[]

for k in input_images:
    a=np.random.randint(input_images.shape[2]-1)
    b=np.random.randint(input_images.shape[1]-1)

    xdim=(a,a+1)
    ydim=(b,b+1)
   
    hotregion_image.append( u.hotregion(k,xdim,ydim))
    deadregion_image.append(u.killregion(k,xdim,ydim))
       
    
hotregion_image=np.array(hotregion_image)
deadregion_image=np.array(deadregion_image)

print "Shape of regular image is: ",input_images.shape
print "Shape of dead image is: " ,deadregion_image.shape
print "Shape of hot image is: " ,hotregion_image.shape

print ''




sample= np.append(data_sample,hotregion_image,axis=0)
sample= np.append(sample,deadregion_image,axis=0)


print sample.shape


X=np.copy(sample)


print "X shape is: ",X.shape

img_rows, img_cols = X.shape[1],X.shape[2]
k=0
for img in X:
    X[k] = X[k] / np.max(img)
    k=k+1



Xtrain= X[:700]
Xtest=X[700:]


print 'Xtrain.shape',Xtrain.shape
print 'Xtest.shape',Xtest.shape


Xtrain, Xtest,input_shape= u.check_test_and_train_images_format(Xtrain, Xtest, img_rows, img_cols)

print 'Xtrain.shape after if statement',Xtrain.shape
print 'Xtest.shape after if statement',Xtest.shape


#from sklearn.model_selection import train_test_split
#train_X,valid_X,train_ground,valid_ground = train_test_split(Xtrain, Xtrain, test_size=0.2, random_state=13)

autoencoder= load_model('empty_AE.hdf5',compile=True)

autoencoder.summary()


checkpointer = ModelCheckpoint(filepath='AE_best_model.hdf5', verbose=2, save_best_only=True)
earlystop= EarlyStopping(monitor='val_loss', min_delta=0, patience=10 ,verbose=1, mode='auto')
autoencoder.fit(Xtrain,Xtrain,epochs=200
                             ,verbose=2
                             ,validation_data=(Xtest[:500],Xtest[:500])
                             ,shuffle=True
                             ,callbacks=[checkpointer,earlystop])

best_model= load_model('AE_best_model.hdf5')
print '\n Finished Training'