import numpy as np
import itertools
import pickle
import os
from keras import backend as K
import h5py
from keras import callbacks
import random
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#helper fucntions
#def get_data(file_name,group='EBOccupancyTask_EBOT_rec_hit_occupancy',data_type='good_2016',preprocess_level=0):
#   "picks samples out of a hdf file and return a numpy array"
#   data_folder=os.environ["DATA"].replace("good_2016",data_type)
#   input_file=h5py.File(data_folder+"/"+file_name,'r')
#   logging.debug("Loading data from file: "+file_name)
#   ret_array=np.array((input_file[group]))
#   ret_array=preprocess(ret_array,preprocess_level)
#   logging.debug("Supplying "+str(ret_array.shape[0])+" samples")
#   return ret_array

def killregion(image,xdim,ydim):  
    tempX=image.copy()
    x1=np.min(xdim)
    x2=np.max(xdim)
    y1=np.min(ydim)
    y2=np.max(ydim)
    if x1==x2 | y1==y2:
        print "error, no change is made on image"
    #if tempX.ndim ==3:
    #    
    #    for k in range(tempX.shape[0]):
    #        
    #        for i in range(x1,x2):
    #            
    #            for j in range(y1,y2):
    #                
    #                tempX[k,j,i]=0
    #
    #else:
        
    for i in range(x1,x2):
            
        for j in range(y1,y2):
                
            tempX[j,i]=0
                
    return tempX


#help (np.zeros)
def hotregion(image,xdim,ydim):  
    tempX=image.copy()
    x1=np.min(xdim)
    x2=np.max(xdim)
    y1=np.min(ydim)
    y2=np.max(ydim)
    if x1==x2 | y1==y2:
        print "error, no change is made on image"
   # if tempX.ndim ==3:
   #     
   #     for k in range(tempX.shape[0]):
   #         
   #         for i in range(x1,x2):
   #             
   #             for j in range(y1,y2):
   #                 
   #                 tempX[k,j,i]=np.max(image)
   # else:
        
    for i in range(x1,x2):
            
        for j in range(y1,y2):
                
            tempX[j,i]=1e4                
    return tempX

def randomregion(image,xdim,ydim):
    
    
    tempX=image.copy()
    x1=abs(np.min(xdim))
    x2=np.max(xdim)
    y1=abs(np.min(ydim))
    y2=np.max(ydim)
    if x1==x2 | y1==y2:
        print "error, no change is made on image"
    #print "x1,x2,y1,y2",x1,x2,y1,y2
    random_noise_region=np.random.randint(np.max(image), size=( abs(x2-x1), abs(y2-y1) ) )   
    #print "random_noise_region.shape",random_noise_region.shape
    

  #  if tempX.ndim == 3:
  #      
  #      for k in range(tempX.shape[0]):
  #          
  #          for i in range(x1,x2):
  #              
  #              for j in range(y1,y2):
  #                  te = i-x1
  #                  te2= j-y1
  #                  tempX[k,j,i]=random_noise_region[te,te2]
  #  
  #  else:
  #      
    for i in range(x1,x2):
            
        for j in range(y1,y2):
            te = i-x1
            te2= j-y1
            tempX[j,i]=random_noise_region[te,te2]
                

    return tempX


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_loss(data, title,yscale="linear"):     
    """ Plots the training and validation loss yscale can be: linear,log,symlog """
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.plot(data.history["loss"])#, linestyle=line_styles[0], color=color_palette["Indigo"][900], linewidth=3)
    plt.plot(data.history["val_loss"])#, linestyle=line_styles[2], color=color_palette["Teal"][300], linewidth=3)
    plt.legend(["Train", "Validation"])#, loc="upper right", frameon=False)
    plt.yscale(yscale)
    plt.show();
    
    
def plot_acc(data, title,yscale="linear"):
    """ Plots the training and validation accuracy 
    yscale can be: linear,log,symlog  """
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title(title)
    plt.plot(data.history["acc"])#, linestyle=line_styles[0], color=color_palette["Indigo"][900], linewidth=3)
    plt.plot(data.history["val_acc"])#, linestyle=line_styles[2], color=color_palette["Teal"][300], linewidth=3)
    plt.legend(["Train", "Validation"])#, loc="upper right", frameon=False)
    plt.yscale(yscale)
    plt.show();
    
    
def check_test_and_train_images_format(Xtrain,Xtest,img_rows, img_cols):
    
    if K.image_data_format() == 'channels_first':

        Xtrain = Xtrain.reshape(Xtrain.shape[0], 1, img_rows, img_cols)
        Xtest = Xtest.reshape(Xtest.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        Xtrain = Xtrain.reshape(Xtrain.shape[0], img_rows, img_cols, 1)
        Xtest = Xtest.reshape(Xtest.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    return Xtrain,Xtest,input_shape


def makedistancemap(image1,image2):
    d=abs(image1-image2)
    return d
