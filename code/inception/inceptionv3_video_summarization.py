
"""
Video summarization using Inception
Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).
"""
import os
import time
import pickle

import cv2

import numpy as np  

from keras import applications
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model

model_load_path = os.path.abspath('../../models/inception_svm')
test_video_path = os.path.abspath('../../')
video_summary_save_path = os.path.abspath('../../results_summary')

start_time = time.time()

base_model = applications.inception_v3.InceptionV3(include_top=True, weights='imagenet', input_tensor=None,
                                                   input_shape=None, pooling=None, classes=1000)
model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)

#Project Model- Loading the saved Inception models
loaded_model1 = pickle.load(open(f'{model_load_path}/model1_inception_svm.sav', 'rb'))
loaded_model2 = pickle.load(open(f'{model_load_path}/model2_inception_svm.sav', 'rb'))

#Path for the video to be summarized
vidcap = cv2.VideoCapture(f'{test_video_path}')

count = 0
bufferCount = 0

globalWideVideo = []
globalOutVideo = []
globalSixVideo = []
globalNoBallVideo = []
globalNoActionVideo = []
buffer = []

th = 5
buff_th = 250
globalWideCounter = 0
globalOutCounter = 0
globalSixCounter = 0
globalNoBallCounter = 0
globalNoActionCounter = 0

while (True):
    success, img = vidcap.read()
    
    if success:
        bufferCount = bufferCount + 1
        buffer.append(img)
        height, width, layers = img.shape
        size = (width, height)
        count = count + 1
        print ('success')
        img1 = cv2.resize(img, (299, 299))
        x = image.img_to_array(img1)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #Feature Extraction Step
        features = model.predict(x) #Inception V3 Model
        predicted_values = loaded_model1.predict(features.reshape(1, -1)) 
        if predicted_values==2:
            predicted_values_2 = loaded_model2.predict(features.reshape(1, -1))
            choices = {'1':'noball', '2':'out', '3':'six', '4':'wide', '5':'noaction'}
            result = choices.get(np.str(int(predicted_values_2)), 'default')
            if result == 'noball':
                globalNoBallCounter = globalNoBallCounter + 1
                print('noball:')
            if result == 'out':
                globalOutCounter = globalOutCounter + 1
                print('out:')
            if result == 'six':
                globalSixCounter = globalSixCounter + 1
                print('six:')
            if result == 'wide':
                globalWideCounter = globalWideCounter + 1
                print('wide:')
            if result == 'noaction':
                globalNoActionCounter = globalNoActionCounter + 1
                print('noaction:')
                
    else:
        break
    ## Frame accumulation
    if bufferCount == buff_th:
        actionCount = {'noball': globalNoBallCounter, 'out': globalOutCounter, 
                       'six': globalSixCounter, 'wide': globalWideCounter}

        winner = max(actionCount, key=actionCount.get)
        if winner == 'noball' and globalNoBallCounter >th:
            globalNoBallVideo.append(buffer)
        if winner == 'out'and globalOutCounter >th:
            globalOutVideo.append(buffer) 
        if winner == 'six'and globalSixCounter >th:
            globalSixVideo.append(buffer)
        if winner == 'wide' and globalWideCounter >th:
            globalWideVideo.append(buffer)
       
        bufferCount = 0
        globalWideCounter = 0
        globalOutCounter = 0
        globalSixCounter = 0
        globalNoBallCounter = 0
        globalNoActionCounter = 0
        buffer = []
    
actionCount = {'noball': globalNoBallCounter, 'out': globalOutCounter,
               'six': globalSixCounter, 'wide': globalWideCounter}
winner = max(actionCount, key=actionCount.get)
if winner == 'noball' and globalNoBallCounter >th:
    globalNoBallVideo.append(buffer)
if winner == 'out'and globalOutCounter >th:
    globalOutVideo.append(buffer) 
if winner == 'six'and globalSixCounter >th:
    globalSixVideo.append(buffer)
if winner == 'wide' and globalWideCounter >th:
    globalWideVideo.append(buffer)

cv2.destroyAllWindows()

print ('Summarizing Video...')

if globalNoBallVideo!=[]:
    noBallVideo = cv2.VideoWriter(f'{video_summary_save_path}/no_ball.avi', cv2.VideoWriter_fourcc(*'DIVX'),
                                  25, size)
    for i in range(len(globalNoBallVideo)):
        for j in range(len(globalNoBallVideo[i])):
            # writing to a image array
            noBallVideo.write(globalNoBallVideo[i][j])
    noBallVideo.release()

if globalOutVideo!=[]:
    outVideo = cv2.VideoWriter(f'{video_summary_save_path}/out.avi', cv2.VideoWriter_fourcc(*'DIVX'),
                               25, size)
    for i in range(len(globalOutVideo)):
        for j in range(len(globalOutVideo[i])):
            # writing to a image array
            outVideo.write(globalOutVideo[i][j])
    outVideo.release()    

if globalWideVideo!=[]:
    wideVideo = cv2.VideoWriter(f'{video_summary_save_path}/wide.avi', cv2.VideoWriter_fourcc(*'DIVX'),
                                25, size)
    for i in range(len(globalWideVideo)):
        for j in range(len(globalWideVideo[i])):
            # writing to a image array
            wideVideo.write(globalWideVideo[i][j])
    wideVideo.release()

if globalSixVideo!=[]:
    sixVideo = cv2.VideoWriter(f'{video_summary_save_path}/sixes.avi', cv2.VideoWriter_fourcc(*'DIVX'),
                               25, size)
    for i in range(len(globalSixVideo)):
        for j in range(len(globalSixVideo[i])):
            # writing to a image array
            sixVideo.write(globalSixVideo[i][j])
    sixVideo.release()
    
print('--- %s seconds ---' % (time.time() - start_time))


