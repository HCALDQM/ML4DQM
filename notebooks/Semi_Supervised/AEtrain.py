import tensorflow as tf
import utils_for_train as u
import h5py
import os
import numpy as np
import random as rn

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

#for reproducibility in Python,numpy and Tensorflow we set their respective seeds as follows
os.environ['PYTHONHASHSEED']='0'
np.random.seed(1)
rn.seed(2)
tf.set_random_seed(3)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

#Force Tensorflow to use a single thread (this is recommended because it might be a cause of randomness)
sess=tf.Session(graph=tf.get_default_graph(),config=session_conf)
K.set_session(sess)


#loading the data

data_folder = '../../data'
file_name = 'HCAL_digi+rechit_occ.hdf5'
group = 'DigiTask_Occupancy_depth_depth1'
input_file=h5py.File(data_folder+"/"+file_name,'r+')

data_sample= np.array(input_file[group])

print data_sample.shape

data_sample=data_sample[:,:,26:58]

print data_sample.shape

input_images=data_sample[:]

#creating artificial bad data images
hotregion_image=[]
deadregion_image=[]

for k in input_images:
    a=np.random.randint(input_images.shape[2]-5)
    b=np.random.randint(input_images.shape[1]-5)

    xdim=(a,a+5)
    ydim=(b,b+5)
   
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


print 'All images together in the sample: ',sample.shape


X=np.copy(sample)
#y=np.zeros((sample.shape[0],1))
Y=[]
yw=['good','hot','dead']



for i in range(3):
    for k in range(input_images.shape[0]):
        Y.append(yw[i])

print "X shape is: ",X.shape

img_rows, img_cols = X.shape[1],X.shape[2]

k=0
maxx=np.max(X[:999])
for img in X:
    X[k] = X[k] / np.max(img)
    k=k+1

Xtrain = X[:700]
#ytrain = y[:700]
ytrain = Y[:700]

Xt =  X[700:]
#yt =  y[700:]
Yt = Y[700:]
#Xval, Xtest ,yval, ytest =train_test_split(Xt,yt,test_size=.4,random_state =5 )

Xval, Xtest ,yval, ytest =train_test_split(Xt,Yt,train_size=.3,random_state =5 )
 
print 'Xtrain.shape',Xtrain.shape
print 'ytrain.shape',len(ytrain)#ytrain.shape
print 'Xval.shape',Xval.shape
print 'yval.shape',len(yval)#yval.shape
print 'Xtest.shape',Xtest.shape
print 'ytest.shape',len(ytest)#ytest.shape


Xtrain, Xval,input_shape= u.check_test_and_train_images_format(Xtrain, Xval, img_rows, img_cols)

#same as last line but for just one set of the data
if K.image_data_format() == 'channels_first':
        Xtest = Xtest.reshape(Xtest.shape[0], 1, img_rows, img_cols)
else:
        Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols, 1)

print 'Xtrain.shape after if statement',Xtrain.shape
print 'ytrain.shape after if statement',len(ytrain)#ytrain.shape
print 'Xval.shape after if statement ', Xval.shape
print 'yval.shape after if statement',  len(yval)#yval.shape
print 'Xtest.shape after if statement', Xtest.shape
print 'ytest.shape after if statement', len(ytest)#ytest.shape

print 'input_shape after if statement', input_shape


#"""
#print 'Xtrain.shape',Xtrain.shape
#print 'Xtest.shape',Xtest.shape
#
#
#Xtrain, Xtest,input_shape= u.check_test_and_train_images_format(Xtrain, Xtest, img_rows, img_cols)
#
#print 'Xtrain.shape after if statement',Xtrain.shape
#print 'Xtest.shape after if statement',Xtest.shape
#
#
##from sklearn.model_selection import train_test_split
##train_X,valid_X,train_ground,valid_ground = train_test_split(Xtrain, Xtrain, test_size=0.2, random_state=13)
#"""


autoencoder= load_model('empty_AE.hdf5',compile=True)
autoencoder.summary()

#tensorboard=keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=50, 
#                            write_graph=True,write_grads=True, write_images=True, 
#                            embeddings_freq=0, embeddings_layer_names=None,embeddings_metadata=None)

checkpointer = ModelCheckpoint(filepath='AE_best_model.hdf5', verbose=2, save_best_only=True)
earlystop= EarlyStopping(monitor='val_loss', min_delta=0, patience=10 ,verbose=1, mode='auto')


autoencoder.fit(Xtrain,Xtrain,epochs=200,batch_size=100
                             ,verbose=1
                             ,validation_data=(Xval,Xval)
                             ,shuffle=True
                             ,callbacks=[checkpointer,earlystop])


print '\n Finished Training'