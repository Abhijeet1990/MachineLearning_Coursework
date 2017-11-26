'''
Author: Abhijeet Sahu

Programming Assignment 2
Machine Learning with Networks
3.
This block of code contains the implementation of Linear and RBF SVM. The impact of Bandwidth(C) and Tau(only for the radial basis function) on error and Number of Support vectors is computed

The Scikit library has been used for this task
'''
import os,sys
import numpy as np
from string import split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as pt


fd1=os.open("bclass",os.O_RDONLY)
os.fchdir(fd1)
for filename in os.listdir(os.getcwd() ):
	if(filename.endswith("-train")):
		f = open(filename, 'r')
		content=f.readlines()
		my_data = [[float(val) for val in line.split()] for line in content[1:]]
		my_array=np.array(my_data)
		Train_Labels=my_array[:,0]
		Train_Feature = my_array[:,1:]

for i in range(len(Train_Labels)):
		if Train_Labels[i]<0.0:
			Train_Labels[i]=0.0

for filename in os.listdir(os.getcwd() ):
	if(filename.endswith("-test")):
		e = open(filename, 'r')
		content1=e.readlines()
		my_data1 = [[float(val) for val in line.split()] for line in content1[1:]]
		my_array1=np.array(my_data1)
		Test_Labels=my_array1[:,0]
		Test_Feature = my_array1[:,1:]
for i in range(len(Test_Labels)):
		if Test_Labels[i]<0.0:
			Test_Labels[i]=0.0

###################Linear SVM#######################################################
MSE_test_data=[]
MSE_dev_data=[]
support_vectors=[]
C=np.array([0.25,0.5,1,2,4])
for j in range(len(C)):
  clf = svm.SVC(C=C[j],kernel='linear')
  clf.fit(Train_Feature, Train_Labels)

  print "Number of Support Vectors"
  print clf.support_vectors_.shape[0]
  support_vectors.append(clf.support_vectors_.shape[0])
  print "Training Data Linear"

  MSE_train=np.mean((clf.predict(Train_Feature)-Train_Labels)**2)
  print MSE_train

  dev_accuracy=cross_val_score(clf,Train_Feature,Train_Labels,cv=5)
  print "Development Data Linear"
  MSE_dev=1-dev_accuracy.mean()
  MSE_dev_data.append(MSE_dev)
  print MSE_dev

  print "Test Data Linear"
  MSE_test=np.mean((clf.predict(Test_Feature)-Test_Labels)**2)
  MSE_test_data.append(MSE_test)
  print MSE_test

pt.plot(C,MSE_test_data,'r',C,MSE_dev_data,'g',linewidth=2.0)
pt.axis([0,4,0.00,0.2])
pt.xlabel('C(Bandwidth)')
pt.ylabel('Mean Square Error')
pt.legend(['Testing Error','Development Error'],loc='upper right')
pt.grid(True)  
pt.show()
pt.plot(C,support_vectors,'r',linewidth=2.0)
pt.axis([0,4,0.00,200])
pt.xlabel('C(Bandwidth)')
pt.ylabel('No. of Vectors')
pt.grid(True)  
pt.show()

##################RBF SVM#########################################################
Tau=np.array([0.25,0.5,1,2,4])
C2=np.array([0.25,0.5,1,2,4])
for i in range(len(Tau)):
  MSE_test_data1=[]
  MSE_dev_data1=[]
  support_vectors1=[]
  for j in range(len(C2)): 
     clf2 = svm.SVC(C=C2[j],kernel='rbf',gamma=Tau[i])
     clf2.fit(Train_Feature, Train_Labels)
     
     print "No. of Support Vectors using RBF"
     print clf2.support_vectors_.shape[0]
     support_vectors1.append(clf2.support_vectors_.shape[0])
     print "Training Data with RBF"
     MSE_train1=np.mean((clf2.predict(Train_Feature)-Train_Labels)**2)
     print MSE_train1

     dev_accuracy=cross_val_score(clf2,Train_Feature,Train_Labels,cv=5)
     print "Development Data with RBF"
     MSE_dev1=1-dev_accuracy.mean()
     MSE_dev_data1.append(MSE_dev1)
     print MSE_dev1

     print "Test Data with RBF"

     MSE_test2=np.mean((clf2.predict(Test_Feature)-Test_Labels)**2)
     MSE_test_data1.append(MSE_test2)
     print MSE_test2
  pt.plot(C2,MSE_test_data1,'r',C2,MSE_dev_data1,'g',linewidth=2.0)
  pt.axis([0,4,0.00,1])
  pt.xlabel('C(Bandwidth)')
  pt.ylabel('Mean Square Error')
  pt.legend(['Testing Error','Development Error'],loc='upper right')
  pt.grid(True)  
  pt.show()
  pt.plot(C2,support_vectors1,'r',linewidth=2.0)
  pt.axis([0,4,0.00,200])
  pt.xlabel('C(Bandwidth)')
  pt.ylabel('No. of Vectors')
  pt.grid(True)  
  pt.show()











	



