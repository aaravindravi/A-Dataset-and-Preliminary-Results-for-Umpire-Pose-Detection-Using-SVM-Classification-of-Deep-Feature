
"""
To extract features for model 2 using VGG19 net 
Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).

"""
import os
import time
import glob

import numpy as np

from keras import applications
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras import applications
from keras.models import Model

data_path = os.path.abspath('../../data/umpire_poses_train')
save_path = os.path.abspath('../../features/vgg19')
classes = ['no_ball', 'out', 'sixes', 'wide', 'no_action']

base_model = applications.vgg19.VGG19(include_top=True, weights='imagenet', input_tensor=None,
                                      input_shape=None, pooling=None, classes=1000)

start_time = time.time()
layers_to_extract = ['fc1']

for layer_num in range(0, len(layers_to_extract)):
    model = Model(input=base_model.input, output=base_model.get_layer(layers_to_extract[layer_num]).output)
    for cls in range(1, 6):
        img_count = 0
        feats=[]
        for image_path in glob.glob(f'{data_path}/{classes[cls-1]}*'):
            img_count = img_count + 1
            print(img_count)
            
            #Pre-processing
            img = image.load_img(image_path, target_size=(224, 224))
            x_in = image.img_to_array(img)
            x_in = np.expand_dims(x_in, axis=0)
            x_in = preprocess_input(x_in)
            
            #Feature Extraction
            features = model.predict(x_in)
            features = features.flatten()
            feats.append(features)
            features_arr = np.char.mod('%f', features)
        
        feature_list = np.squeeze(np.asarray(feats))
        labels = np.ones(len(feature_list))*cls
        feature_list = np.column_stack((feature_list, labels))
        
        np.save(f'{save_path}/model2_vgg19_{layers_to_extract[layer_num]}_features_{classes[cls-1]}.npy',
                feature_list)

print("--- %s seconds ---" % (time.time() - start_time))
    