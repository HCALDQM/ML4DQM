import numpy as np
import pickle
import os
import h5py
from keras import callbacks
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#helper fucntions
def get_data(file_name,group='EBOccupancyTask_EBOT_rec_hit_occupancy',data_type='good_2016',preprocess_level=0):
   "picks samples out of a hdf file and return a numpy array"
   data_folder=os.environ["DATA"].replace("good_2016",data_type)
   input_file=h5py.File(data_folder+"/"+file_name,'r')
   logging.debug("Loading data from file: "+file_name)
   ret_array=np.array((input_file[group]))
   ret_array=preprocess(ret_array,preprocess_level)
   logging.debug("Supplying "+str(ret_array.shape[0])+" samples")
   return ret_array
