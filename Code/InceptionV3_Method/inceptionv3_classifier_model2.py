
"""
Svm classifier for model 2 using inception
Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).

"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut
import pickle
import time


#Storing the extracted features
X=[]
X1 = np.load('class_1data.npy')
X2 = np.load('class_2data.npy')
X3 = np.load('class_3data.npy')
X4 = np.load('class_4data.npy')
X5 = np.load('class_5data.npy')

X_data = np.append(X1,X2,axis=0)
X_data = np.append(X_data,X3,axis=0)
X_data = np.append(X_data,X4,axis=0)
X_data = np.append(X_data,X5,axis=0)

Y_data = X_data[:,2048]
X_data = X_data[:,0:2048]

#Split the data into 80% training and 20% test data
x_tr,x_ts,y_tr,y_ts = train_test_split(X_data, Y_data, test_size=0.2,random_state=26)

#Classifier using Linear SVM

clf = LinearSVC(C=10)
start_time = time.time()
clf = clf.fit(x_tr,y_tr)
predictions_tr = (clf.predict(x_ts))

#10-fold Cross-Validation    
scores = cross_val_score(clf, x_tr, y_tr, cv=10)
test_acc = accuracy_score(y_ts,predictions_tr)

print("Training Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Test Accuracy: %0.4f" % test_acc)
print("--- %s seconds ---" % (time.time() - start_time))


##Leave One Out Validation
loo_train_acc=[]
loo = LeaveOneOut()
for train_index, test_index in loo.split(x_tr):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = x_tr[train_index], x_tr[test_index]
   y_train, y_test = y_tr[train_index], y_tr[test_index]
   clf = clf.fit(X_train,y_train)
   predictions = (clf.predict(X_test))
   loo_train_acc.append(accuracy_score(y_test,predictions))

loo_train_accuracy = np.asarray(loo_train_acc)
print("LOO Accuracy: %0.4f" % loo_train_accuracy.mean())

#Save the model
pickle.dump(clf, open('cricket_inceptionv3_equal_linsvm.sav', 'wb'))

