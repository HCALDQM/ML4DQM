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
from keras.layers import Dense, Dropout, Flatten,Conv2D, MaxPooling2D,Activation,BatchNormalization,LeakyReLU
from keras.utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam

data_folder = '../data'
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

#im=plt.imshow(input_images[1],cmap=cm.coolwarm)
#plt.show()
#plt.clf()
#
#im=plt.imshow(hotregion_image[1],cmap=cm.coolwarm)
#plt.show()
#plt.clf()#plt.clf clears the figure and it's axis but leaves the window open. 
#         #as opposed to plt.close which closes the window. If you are showing many images at a time without
#         #closing the window it is better.
#
#im=plt.imshow(deadregion_image[1],cmap=cm.coolwarm)
#plt.show()
#plt.clf()
#input_image=np.reshape(input_image,(input_image.shape[0],input_image.shape[1]))




sample= np.append(data_sample,hotregion_image,axis=0)
sample= np.append(sample,deadregion_image,axis=0)


print sample.shape


X=np.copy(sample)

y=np.zeros((sample.shape[0],1))




#this is because I am only going to add the hot and dead region images

for i in range(2,0,-1):
    y[-i*input_images.shape[0]:]=3-i
#if you want it as intergers leave this as it is
#if you would like it as a vector then do this
y=to_categorical(y,3)

print "X shape is: ",X.shape
print "y shape is: ",y.shape
# input image dimensions
img_rows, img_cols = X.shape[1],X.shape[2]



Xtrain, Xtest ,ytrain, ytest =train_test_split(X,y,test_size=.4,random_state =5 )

print 'Xtrain.shape',Xtrain.shape
print 'Xtest.shape',Xtest.shape
print 'ytrain.shape',ytrain.shape
print 'ytest.shape',ytest.shape
print 'Printing labels and their corresponding images'
    
#for k in range(10):
#    print'-------------------'
#    print ytrain[k]
#    im=plt.imshow(Xtrain[k],cmap=cm.gray)
#    plt.show()
#    plt.clf()


Xtrain, Xtest,input_shape= u.check_test_and_train_images_format(Xtrain, Xtest, img_rows, img_cols)

print 'Xtrain.shape after if statement',Xtrain.shape
print 'Xtest.shape after if statement',Xtest.shape
print 'ytrain.shape after if statement',ytrain.shape
print 'ytest.shape after if statement',ytest.shape
    

###################
#dummy model

empty_model= load_model('emptymodel.hdf5')

###################




#############################################
#best model so far
#model = Sequential([
#Conv2D(10, kernel_size=(2, 2), activation='relu', strides=(1, 1),input_shape=input_shape),
#MaxPooling2D(pool_size=(2,2)),
#BatchNormalization(),
#Conv2D(8, kernel_size=(3, 3), activation='relu', strides=(1, 1)),
#MaxPooling2D(pool_size=(2,2)),
#Conv2D(8,kernel_size=(1,1), activation='relu'),
#Dropout(0.25),
#Flatten(),
#
#Dense(8,activation='relu'),
#Dense(3, activation='softmax')
#])
#

############################################



############################################
#this is my model
#model = Sequential([
#BatchNormalization(input_shape=input_shape),
#Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
#Conv2D(8, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
#Dropout(0.25),
#Flatten(),
#Dense(3, activation='softmax')
#])
############################################

#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',#Adam(lr=1e-3),
#              metrics=['accuracy'])

empty_model.summary()


checkpointer = ModelCheckpoint(filepath='best_trained_model_good_hot_dead.hdf5', verbose=0, save_best_only=True)
earlystop= EarlyStopping(monitor='val_loss', min_delta=0, patience=200 ,verbose=1, mode='auto')

history = empty_model.fit(Xtrain,ytrain,epochs=100
                    ,verbose=0
                    ,validation_data=(Xtest,ytest)
                    ,shuffle=False
                    ,callbacks=[checkpointer])#,earlystop])



best_model= load_model('best_trained_model_good_hot_dead.hdf5')
best_model.summary()

ypred=best_model.predict(Xtest)
ypredproba=best_model.predict_proba(Xtest)
#it's preferable to use .predict_classes because .predict might give probabilities 
#and not the label's in the case of multiclass
ypredclass=best_model.predict_classes(Xtest)

ypredclass=np.reshape(ypredclass,(ypredclass.shape[0],1))
#use argmax(1) to give the position of max value in a categorical variable like ytest
#it's basically a .predict_classes for ytest
ytestclass= ytest.argmax(1)
ytestclass=np.reshape(ytestclass,(ytestclass.shape[0],1))


confusion= metrics.confusion_matrix(ytestclass,ypredclass)

print confusion

fpr,tpr,thresholds= roc_curve(ytestclass,ypredproba[:,0],pos_label=0)
roc_auc = auc(fpr, tpr)
print roc_auc,'For the label #0'