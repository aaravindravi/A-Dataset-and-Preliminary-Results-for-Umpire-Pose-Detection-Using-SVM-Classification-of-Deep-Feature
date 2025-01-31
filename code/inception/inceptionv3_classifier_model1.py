
"""
SVM classifier for model 1 using inception
Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).

"""
import pickle
import time
import os

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut

feature_path = os.path.abspath('../../features/inception')
model_save_path = os.path.abspath('../../models/inception_svm')
leave_one_out_validation = False

#Storing the extracted features
non_umpire = np.load(f'{feature_path}/model1_inception_features_non_umpire.npy')
umpire = np.load(f'{feature_path}/model1_inception_features_umpire.npy')

X_data = np.append(non_umpire, umpire, axis=0)
labels = X_data[:, (X_data.shape[1]-1)]
train_data = X_data[:, 0:(X_data.shape[1]-1)]

x_tr, x_ts, y_tr, y_ts = train_test_split(train_data, labels, test_size=0.2, random_state=10)

#Classifier using Linear SVM
clf = LinearSVC(C=10)
start_time = time.time()
clf = clf.fit(x_tr, y_tr)
predictions_tr = (clf.predict(x_ts))

#10-fold Cross-Validation    
scores = cross_val_score(clf, x_tr, y_tr, cv=10)
test_acc = accuracy_score(y_ts, predictions_tr)

print("Training Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Test Accuracy: %0.4f" % test_acc)
print("--- %s seconds ---" % (time.time() - start_time))

#Leave One Out Validation
if leave_one_out_validation:
   loo_train_acc = []
   loo = LeaveOneOut()
   for train_index, test_index in loo.split(x_tr):
      X_train, X_test = x_tr[train_index], x_tr[test_index]
      y_train, y_test = y_tr[train_index], y_tr[test_index]
      clf = clf.fit(X_train, y_train)
      predictions = (clf.predict(X_test))
      loo_train_acc.append(accuracy_score(y_test, predictions))

   loo_train_accuracy = np.asarray(loo_train_acc)
   print("LOO Accuracy: %0.4f" % loo_train_accuracy.mean())

#Save the model
pickle.dump(clf, open(f'{model_save_path}/model1_inception_svm.sav', 'wb'))
