
"""
Feature extraction for model2 using Inception
Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).

"""
import os
import glob
import time

import numpy as np

from keras import applications
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras import applications
from keras.models import Model

data_path = os.path.abspath('../../data/umpire_poses_train')
save_path = os.path.abspath('../../features/inception')
classes = ['no_ball', 'out', 'sixes', 'wide', 'no_action']

#Download the base model as Inception V3
base_model = applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)

start_time = time.time()

for cls in range(1, 6):
    img_count = 0
    feats = []
    for image_path in glob.glob(f'{data_path}/{classes[cls-1]}*'):
        img_count = img_count + 1
        print(img_count)
        
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        features = model.predict(x)
        feats.append(features)
        features_arr = np.char.mod('%f', features)
    
    feature_list = np.squeeze(np.asarray(feats))
    
    labels = np.ones(len(feature_list))*cls
    feature_list = np.column_stack((feature_list, labels))
    
    np.save(f'{save_path}/model2_inception_features_{classes[cls-1]}.npy', feature_list)
     
print("--- %s seconds ---" % (time.time() - start_time))
     
