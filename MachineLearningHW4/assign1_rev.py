'''
Author: Abhijeet Sahu

Programming Assignment 2
Machine Learning with Networks
1.

This block of code contains the implementation of Logisitic Regression for predicting Parameters using Gradient Descent method and finding Training and Testing Error

The Scikit implementation has also been done to compare the performance of my own function with the SCIKIT implementation


'''
import os,sys
import numpy as np
from string import split
from sklearn import linear_model
from sklearn import preprocessing

class LogisticRegressionSelf:

    def probability(self):
      product=self.features.dot(self.w)
      expr=np.exp(-product)
      return 1/(1+expr)

    def log_likelihood(self):
      p = self.probability()
      #Get Log Likelihood For Each Row of Dataset
      loglikelihood = self.labels*np.log(p+1e-24) + (1-self.labels)*np.log(1-p+1e-24)
      return -1*loglikelihood.sum()

    def log_likelihood_gradient(self):
      error = self.labels-self.probability()
      product = error*self.features
      grad = product.sum(axis=0).reshape(self.w.shape)
      return grad


    def gradient_decent(self,alpha=1e-2,max_iterations=1e4):
      previous_likelihood = self.log_likelihood()
      difference = self.tolerance+1
      iteration = 0
      self.likelihood_history = [previous_likelihood]
      while (difference > self.tolerance) and (iteration < max_iterations):
        self.w = self.w + alpha*self.log_likelihood_gradient()
        temp = self.log_likelihood()
        difference = np.abs(temp-previous_likelihood)
        previous_likelihood = temp
        self.likelihood_history.append(previous_likelihood)
        iteration += 1
	


    def __init__(self,X,y,tolerance=1e-5):
        self.tolerance = tolerance
        self.labels = y.reshape(y.size,1)
        self.w = np.zeros((X.shape[1],1))
        self.features = np.ones((X.shape[0],X.shape[1]))
        self.features = X
        self.likelihood_history = []
	


itr=100 #this simulation runs for 100 number of iterations
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

log = LogisticRegressionSelf(Train_Feature,Train_Labels,tolerance=1e-6)
log.gradient_decent(alpha=1e-2,max_iterations=itr)
product=log.w.T.dot(Train_Feature.T).T
expr=np.exp(-product)
prob= 1/(1+expr)
for i in range(len(prob)):
	if prob[i]>0.5:
		prob[i]=1.0
	else:
		prob[i]=0.0

probr=prob.reshape(Train_Labels.shape[0])
MSE_train1=np.mean((probr-Train_Labels)**2)

#print probr
#print Train_Labels
print "Training error with Raw data"
print MSE_train1
#################################Find L1 normalized Train Data##############################
Train_L1normalized = preprocessing.normalize(Train_Feature, norm='l1')
#################################Find L2 normalized Train Data##############################
Train_L2normalized = preprocessing.normalize(Train_Feature, norm='l2')

logl1 = LogisticRegressionSelf(Train_L1normalized,Train_Labels,tolerance=1e-6)
logl1.gradient_decent(alpha=1e-2,max_iterations=itr)
product1=logl1.w.T.dot(Train_L1normalized.T).T
expr=np.exp(-product1)
probl1= 1/(1+expr)
for i in range(len(probl1)):
	if probl1[i]>0.5:
		probl1[i]=1.0
	else:
		probl1[i]=0.0
probl1r=probl1.reshape(Train_Labels.shape[0])
MSE_trainl1=np.mean((probl1r-Train_Labels)**2)
print "Training error with L1 norm"
print MSE_trainl1

logl2 = LogisticRegressionSelf(Train_L2normalized,Train_Labels,tolerance=1e-6)
logl2.gradient_decent(alpha=1e-2,max_iterations=itr)
product2=logl2.w.T.dot(Train_L2normalized.T).T
expr=np.exp(-product2)
probl2= 1/(1+expr)
for i in range(len(probl2)):
	if probl2[i]>0.5:
		probl2[i]=1.0
	else:
		probl2[i]=0.0
probl2r=probl2.reshape(Train_Labels.shape[0])
MSE_trainl2=np.mean((probl2r-Train_Labels)**2)
print "Training error with L2 norm"
print MSE_trainl2


#C=the maximum number of iterations....we can change the value of C from 1 to 100 and see the performance 
logSKLl1=linear_model.LogisticRegression(tol=1e-6,C=itr)
logSKLl1.fit(Train_L1normalized,Train_Labels)
coefl1 = np.append(logSKLl1.intercept_,logSKLl1.coef_)
templ1 = LogisticRegressionSelf(Train_L1normalized,Train_Labels,tolerance=1e-6)
templ1.w = coefl1

logSKLl2=linear_model.LogisticRegression(tol=1e-6,C=itr)
logSKLl2.fit(Train_L2normalized,Train_Labels)
coefl2 = np.append(logSKLl2.intercept_,logSKLl2.coef_)
templ2 = LogisticRegressionSelf(Train_L2normalized,Train_Labels,tolerance=1e-6)
templ2.w = coefl2

logSKL = linear_model.LogisticRegression(tol=1e-6,C=itr)
logSKL.fit(Train_Feature,Train_Labels)
coef = np.append(logSKL.intercept_,logSKL.coef_)
temp = LogisticRegressionSelf(Train_Feature,Train_Labels,tolerance=1e-6)
temp.w = coef
#print coef.T,temp.log_likelihood()

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

print("test feature")
product_test=log.w.T.dot(Test_Feature.T).T
expr=np.exp(-product_test)
prob1= 1/(1+expr)

for i in range(len(prob1)):
	if prob1[i]>0.5:
		prob1[i]=1.0
	else:
		prob1[i]=0.0

prob1r=prob1.reshape(Test_Labels.shape[0])
MSE_test1=np.mean((prob1r-Test_Labels)**2)
print "Testing error"
print MSE_test1

#################################Find L1 normalized Test Data##############################
Test_L1normalized = preprocessing.normalize(Test_Feature, norm='l1')
#################################Find L2 normalized Test Data##############################
Test_L2normalized = preprocessing.normalize(Test_Feature, norm='l2')

product_testl1=logl1.w.T.dot(Test_L1normalized.T).T
expr=np.exp(-product_testl1)
probl1= 1/(1+expr)
for i in range(len(probl1)):
	if probl1[i]>0.5:
		probl1[i]=1.0
	else:
		probl1[i]=0.0

probl1r=probl1.reshape(Test_Labels.shape[0])
MSE_testl1=np.mean((probl1r-Test_Labels)**2)
print "Testing error with L1 norm"
print MSE_testl1

product_testl2=logl2.w.T.dot(Test_L2normalized.T).T
expr=np.exp(-product_testl2)
probl2= 1/(1+expr)
for i in range(len(probl2)):
	if probl2[i]>0.5:
		probl2[i]=1.0
	else:
		probl2[i]=0.0
probl2r=probl2.reshape(Test_Labels.shape[0])
MSE_testl2=np.mean((probl2r-Test_Labels)**2)
print "Testing error with L2 norm"
print MSE_testl2

print "###########SCIKIT implementation####################"
#prediction for training raw data using SCIKIT toolkit
MSE_train=np.mean((logSKL.predict(Train_Feature)-Train_Labels)**2)
print "Training Error for Raw Data"
print MSE_train

MSE_trainl1=np.mean((logSKLl1.predict(Train_L1normalized)-Train_Labels)**2)
print "Training Error for L1 normalized Data"
print MSE_trainl1

#prediction for training L2 normalized data using SCIKIT toolkit
MSE_trainl2=np.mean((logSKLl2.predict(Train_L2normalized)-Train_Labels)**2)
print "Training Error for L2 normalized Data"
print MSE_trainl2

#prediction for test data using SCIKIT toolkit
MSE_test=np.mean((logSKL.predict(Test_Feature)-Test_Labels)**2)
print "Testing Error for Raw Data"
print MSE_test

#prediction for test L1 normalized data using SCIKIT toolkit
MSE_testl1=np.mean((logSKLl1.predict(Test_L1normalized)-Test_Labels)**2)
print "Testing Error for L1 normalized Data"
print MSE_testl1

#prediction for test L2 normalized data using SCIKIT toolkit
MSE_testl2=np.mean((logSKLl2.predict(Test_L2normalized)-Test_Labels)**2)
print "Testing Error for L2 normalized Data"
print MSE_testl2





		




