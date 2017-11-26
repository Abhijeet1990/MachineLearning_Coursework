'''
Author: Abhijeet Sahu

Programming Assignment 2
Machine Learning with Networks
2.

This block of code contains the implementation of Locally weighted Logisitic Regression for predicting Parameters using Newton Method


'''
import os,sys
import numpy as np
from string import split
from sklearn import linear_model
from sklearn import preprocessing
from numpy.linalg import inv
import numpy.matlib
import matplotlib.pyplot as pt

class LogisticRegressionSelf:

    def probability(self):
      product=np.mat(self.features)*self.w
      expr=np.exp(-np.sum(product))
      prob=1/(1+expr)
      print prob
      return prob


    #this function calculates w for each test data
    def weight_calc(self,test_data,tau=5):
      po= np.power((self.features-np.matlib.repmat(test_data.T, self.features.shape[0], 1)),2)
      ye=-np.sum(po,axis=1)/(self.features.shape[1])     
      ra= 2 * (tau ** 2)
      se= np.exp(ye/ra)
      self.weight = np.reshape(se,(self.features.shape[0],1))


    def weighted_log_likelihood(self):
      #Get Probablities
      p = self.probability()
      #Get Log Likelihood For Each Row of Dataset
      loglikelihood = self.weight*(self.labels*np.log(p+1e-24) + (1-self.labels)*np.log(1-p+1e-24))
      loglikelihood = loglikelihood.sum()
      #adding the regularization term
      loglikelihood -= 0.001*np.power(self.w, 2).sum()
      #Return net expression
      return loglikelihood

    def log_likelihood_gradient(self):
      error=np.zeros((self.features.shape[0],1),float)
      for i in range(self.features.shape[0]):
         data=np.mat(self.features[i,:])*self.w
         p=1.0/(1+np.exp(-np.sum(data)))
         error[i] = self.weight[i]*(self.labels[i]-p) 
      product = error*self.features
      grad=product.sum(axis=0).reshape(self.w.shape)
      grad -= 0.001*self.w
      return grad


    def Hessian(self):
      Dii=np.zeros(self.features.shape[0],float)
      for i in range(self.features.shape[0]):
        data2=np.mat(self.features[i,:])*self.w
        p=1.0/(1+np.exp(-np.sum(data2)))
      	Dii[i]= -self.weight[i]*p*(1-p) 
      
      D_i_i =  np.reshape (Dii,self.features.shape[0])
      m=np.diag(D_i_i)
      hess=np.mat(self.features.T)*np.mat(m)*np.mat(self.features) 
      hess -= 0.001*np.identity(self.features.shape[1])
      return hess

    def newton_method(self,max_iter=5):
       iteration=0
       prev=self.Hessian()
       while(iteration < max_iter):
	  self.w = self.w - inv(prev)*self.log_likelihood_gradient()
          temp=self.Hessian()
	  prev=temp
          iteration+=1
     

    def __init__(self,X,y):
        
        self.labels = y.reshape(y.size,1)
        self.w = np.zeros((X.shape[1],1),float)
        self.features = np.ones((X.shape[0],X.shape[1]))
        self.features = X
	self.weight = np.zeros((X.shape[0],1));

test_error=[]
train_error=[]
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


log = LogisticRegressionSelf(Train_Feature,Train_Labels)
Tau=np.array([5.0,1.0,0.5,0.1,0.05,0.01])
prob1=np.zeros(len(Train_Labels))
for j in range(len(Tau)): 
  for i in range(len(Train_Labels)):
    log.w = np.zeros((Train_Feature.shape[1],1),float)
    log.weight = np.zeros((Train_Feature.shape[0],1))
    log.weight_calc(Train_Feature[i,:],Tau[j])
    log.newton_method()
    product_test=Train_Feature*log.w
    prob1[i]= 1.0/(1+np.exp(-np.sum(product_test[i])))
    if prob1[i] > 0.5:
	prob1[i]=1.0
    else:
 	prob1[i]=0.0

  
  MSE_train1=np.mean((prob1-Train_Labels)**2)
  print "Training error"
  print MSE_train1
  print "error%"
  print np.sum(np.absolute(prob1-Train_Labels))/len(Train_Labels)*100
  train_error.append(MSE_train1)

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

Tau=np.array([5.0,1.0,0.5,0.1,0.05,0.01])
prob1=np.zeros(len(Test_Labels))
for j in range(len(Tau)): 
  for i in range(len(Test_Labels)):
    
    log.w = np.zeros((Test_Feature.shape[1],1),float)
    log.weight = np.zeros((Test_Feature.shape[0],1))
    log.weight_calc(Test_Feature[i,:],Tau[j])
    log.newton_method()
    product_test=Test_Feature*log.w
    prob1[i]= 1.0/(1+np.exp(-np.sum(product_test[i])))
    if prob1[i] > 0.5:
	prob1[i]=1.0
    else:
 	prob1[i]=0.0

  
  MSE_test1=np.mean((prob1-Test_Labels)**2)
  print "Testing error"
  print MSE_test1
  print "error%"
  print np.sum(np.absolute(prob1-Test_Labels))/len(Test_Labels)*100
  test_error.append(MSE_test1)

pt.plot(Tau,test_error,'r',Tau,train_error,'g',linewidth=2.0)
pt.axis([0,6,0.00,1.0])
pt.xlabel('Tau')
pt.ylabel('Mean Square Error')
pt.legend(['Testing Error','Training Error'],loc='upper right')
pt.grid(True)  
pt.show()





		




