
"""
SVM Classifier for model 1 using VGG 19 features
Paper: Ravi, Aravind, Harshwin Venugopal, Sruthy Paul, and Hamid R. Tizhoosh. 
"A Dataset and Preliminary Results for Umpire Pose Detection Using SVM Classification of Deep Features." 
arXiv preprint arXiv:1809.06217 (2018).

"""
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
import pickle
import time

#layers_to_extract = ["fc1","fc2"]
layers_to_extract = ["fc1"]

layer_num = 0

#Loading the Features
X1 = np.load("class_cricket_model1"+layers_to_extract[layer_num]+"1vgg19_data.npy")
X2 = np.load("class_cricket_model1"+layers_to_extract[layer_num]+"2vgg19_data.npy")

#Concatenate into single matrix
X_data = np.append(X1,X2,axis=0)

#Labels
Y_data = X_data[:,(X_data.shape[1]-1)]

#Training Data
X_data = X_data[:,0:(X_data.shape[1]-1)]

#Train Test Split 80-20
x_tr,x_ts,y_tr,y_ts = train_test_split(X_data, Y_data, test_size=0.2,random_state=157)

#Classifier SVM Linear Kernel 
clf = LinearSVC(C=10)

start_time = time.time()

clf = clf.fit(x_tr,y_tr)
predictions_tr = (clf.predict(x_ts))

#10-Fold Cross-validation Accuracy
scores = cross_val_score(clf, x_tr, y_tr, cv=10)
print(layers_to_extract[layer_num])
print("Training Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("--- %s seconds ---" % (time.time() - start_time))


#Leave One Out or Jack-Knife Crossvalidation
loo_train_acc=[]
loo = LeaveOneOut()
for train_index, test_index in loo.split(x_tr):
   X_train, X_test = x_tr[train_index], x_tr[test_index]
   y_train, y_test = y_tr[train_index], y_tr[test_index]
   clf = clf.fit(X_train,y_train)
   predictions = (clf.predict(X_test))
   loo_train_acc.append(accuracy_score(y_test,predictions))

loo_train_accuracy = np.asarray(loo_train_acc)
print("LOO Accuracy: %0.4f" % loo_train_accuracy.mean())

#20% Test Data Accuracy
test_acc = accuracy_score(y_ts,predictions_tr)
print("Test Accuracy: %0.4f" % test_acc)

#Save the SVM Model
pickle.dump(clf, open('FER_vgg19fc2_model2net_ck_transfer_only_svm.sav', 'wb'))
