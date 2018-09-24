
"""
Feature extraction for model 1 using Inception
Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).

"""

import os
import numpy as np
import glob
from keras import applications
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras import applications
from keras.models import Model
import time


#Download the base model as Inception V3
base_model = applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)


start_time = time.time()

curr_cwd = os.getcwd() #path to the Dataset folder


#SNOW Dataset 
for cls in range (1,3):
    #Folders containing the dataset
    classes = ["non_umpire","umpire"]

    img_count=0
    feats=[]
    #Extract the output of the avg_pool layer of inception
    
    for image_path in glob.glob(curr_cwd+"/"+classes[cls-1]+"//*"):
        img_count=img_count+1
        print(img_count)
        # load image and set image size to 299 x 299
        img = image.load_img(image_path, target_size=(299, 299))
        # convert image to numpy array
        x = image.img_to_array(img)
        # convert the image into array of shape (3, 299, 299) 
        # need to expand it to (1, 3, 299, 299)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # extracting the features from the given layer
        features = model.predict(x)
        
        feats.append(features)
        features_arr = np.char.mod('%f', features)
    
    feature_list = np.squeeze(np.asarray(feats))
    #storing the labels based on the class number
    labels = np.ones(len(feature_list))*cls
    feature_list = np.column_stack((feature_list,labels))
    #Save the features as a numpy array for further processing
    np.save("class_"+str(cls)+"data_model1.npy",feature_list)
     
     
print("--- %s seconds ---" % (time.time() - start_time))
