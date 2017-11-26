import os,sys
import numpy as np
from string import split
from sklearn import linear_model
from sklearn import preprocessing

class LogisticRegressionSelf:

    def probability(self):
      """Computes the logistic probability of being a positive example
 
    Returns
    -------
    out : ndarray (1,)
        Probablity of being a positive example...sigmoid function
      """
      product=self.features.dot(self.w)
      #print(product)
      expr=np.exp(-product)
      
      return 1/(1+expr)

    def log_likelihood(self):
      """Calculate the loglikelihood for the current set of weights and features.
 
    Returns
    -------
    out : float
      """
      #Get Probablities
      p = self.probability()
      #Get Log Likelihood For Each Row of Dataset
      loglikelihood = self.labels*np.log(p+1e-24) + (1-self.labels)*np.log(1-p+1e-24)
      print("loglikeliood")
      print(loglikelihood.shape)
      #Return Sum
      return -1*loglikelihood.sum()

    def log_likelihood_gradient(self):
      """Calculate the loglikelihood gradient for the current set of weights and features.
 
    Returns
    -------
    out : ndarray(n features, 1)
        gradient of the loglikelihood
      """
      error = self.labels-self.probability()
      
      product = error*self.features
      print "product"
      print(product.shape)
      grad = product.sum(axis=0).reshape(self.w.shape)
      return grad


    def gradient_decent(self,alpha=1e-7,max_iterations=1e4):
      """Runs the gradient decent algorithm
 
    Parameters
    ----------
    alpha : float
        The learning rate for the algorithm
 
    max_iterations : int
        The maximum number of iterations allowed to run before the algorithm terminates
 
      """
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
        """Initializes Class for Logistic Regression
 
        Parameters
        ----------
        X : ndarray(n-rows,m-features)
            Numerical training data.
 
        y: ndarray(n-rows,)
            Interger training labels.
 
        tolerance : float (default 1e-5)
            Stopping threshold difference in the loglikelihood between iterations.
 
        """
        self.tolerance = tolerance
        self.labels = y.reshape(y.size,1)
        #create weights equal to zero with an intercept coefficent at index 0
        self.w = np.zeros((X.shape[1]+1,1))
	print(self.w.shape)
        #Add Intercept Data Point of 1 to each row
        self.features = np.ones((X.shape[0],X.shape[1]+1))
	
        self.features[:,1:] = X
        print("feature shape")
	print(self.features.shape)
        self.shuffled_features = self.features
        self.shuffled_labels = self.labels
	#print(self.labels)
        self.likelihood_history = []
	



fd1=os.open("/home/bo/Desktop/MachineLearningHW4/bclass",os.O_RDONLY)
os.fchdir(fd1)
for filename in os.listdir(os.getcwd() ):
	if(filename.endswith("-train")):
		f = open(filename, 'r')
		content=f.readlines()
		
		# Do a double-nested list comprehension to get the rest of the data into your matrix
		my_data = [[float(val) for val in line.split()] for line in content[1:]]
		#print(my_data)
		my_array=np.array(my_data)
		#print(my_array.shape)#so we have 199 rows and 35 columns...1st column being the label and the rest being label
		Train_Labels=my_array[:,0]
		Train_Feature = my_array[:,1:]

for i in range(len(Train_Labels)):
		if Train_Labels[i]<0.0:
			Train_Labels[i]=0.0		

log = LogisticRegressionSelf(Train_Feature,Train_Labels,tolerance=1e-6)
log.gradient_decent(alpha=2e-2,max_iterations=5e0)
print log.w.T,log.log_likelihood()
'''	
#print "Dimension of w"
Train_predict=log.w.T[:,1:].dot(Train_Feature.T).T
print "code prediction"
print Train_predict

for i in range(len(Train_predict)):
	if Train_predict[i]>0:
		Train_predict[i]=1.0
	else:
		Train_predict[i]=0.0
print Train_predict
print Train_Labels
'''

product=log.w.T[:,1:].dot(Train_Feature.T).T
expr=np.exp(-product)
prob= 1/(1+expr)
#print(prob)
for i in range(len(prob)):
	if prob[i]>0.5:
		prob[i]=1.0
	else:
		prob[i]=0.0

MSE_train1=np.mean((prob-Train_Labels)**2)
print "Training error"
print MSE_train1
#################################Find L1 normalized Train Data##############################
Train_L1normalized = preprocessing.normalize(Train_Feature, norm='l1')
#################################Find L2 normalized Train Data##############################
Train_L2normalized = preprocessing.normalize(Train_Feature, norm='l2')

#C=the maximum number of iterations....we can change the value of C from 1 to 100 and see the performance 
logSKLl1=linear_model.LogisticRegression(tol=1e-6,C=5e0)
logSKLl1.fit(Train_L1normalized,Train_Labels)
coefl1 = np.append(logSKLl1.intercept_,logSKLl1.coef_)
templ1 = LogisticRegressionSelf(Train_L1normalized,Train_Labels,tolerance=1e-6)
templ1.w = coefl1

logSKLl2=linear_model.LogisticRegression(tol=1e-6,C=5e0)
logSKLl2.fit(Train_L2normalized,Train_Labels)
coefl2 = np.append(logSKLl2.intercept_,logSKLl2.coef_)
templ2 = LogisticRegressionSelf(Train_L2normalized,Train_Labels,tolerance=1e-6)
templ2.w = coefl2

logSKL = linear_model.LogisticRegression(tol=1e-6,C=5e0)
logSKL.fit(Train_Feature,Train_Labels)
coef = np.append(logSKL.intercept_,logSKL.coef_)
temp = LogisticRegressionSelf(Train_Feature,Train_Labels,tolerance=1e-6)
temp.w = coef
#print coef.T,temp.log_likelihood()

for filename in os.listdir(os.getcwd() ):
	if(filename.endswith("-test")):
		e = open(filename, 'r')
		content1=e.readlines()
		# Do a double-nested list comprehension to get the rest of the data into your matrix
		my_data1 = [[float(val) for val in line.split()] for line in content1[1:]]
		#print(my_data)
		my_array1=np.array(my_data1)
		#print(my_array.shape)#so we have 199 rows and 35 columns...1st column being the label and the rest being label
		Test_Labels=my_array1[:,0]
		Test_Feature = my_array1[:,1:]
for i in range(len(Test_Labels)):
		if Test_Labels[i]<0.0:
			Test_Labels[i]=0.0

print("test feature")
print(Test_Feature.shape)
product_test=log.w.T[:,1:].dot(Test_Feature.T).T
expr=np.exp(-product_test)
prob1= 1/(1+expr)
#print(prob1)
for i in range(len(prob1)):
	if prob1[i]>0.5:
		prob1[i]=1.0
	else:
		prob1[i]=0.0

MSE_test1=np.mean((prob1-Test_Labels)**2)
print "Testing error"
print MSE_test1
'''
for i in range(len(prob1)):
	print('> predicted=' + repr(prob1[i]) + ', actual=' + repr(Test_Labels[i]))
'''
#################################Find L1 normalized Test Data##############################
Test_L1normalized = preprocessing.normalize(Test_Feature, norm='l1')
#################################Find L2 normalized Test Data##############################
Test_L2normalized = preprocessing.normalize(Test_Feature, norm='l2')

#prediction for training raw data using SCIKIT toolkit
MSE_train=np.mean((logSKL.predict(Train_Feature)-Train_Labels)**2)
print MSE_train
#print(logSKL.predict(Train_Feature))
#print(Train_Labels)
#prediction for training L1 normalized data using SCIKIT toolkit
MSE_trainl1=np.mean((logSKLl1.predict(Train_L1normalized)-Train_Labels)**2)
print MSE_trainl1
#print(logSKLl1.predict(Train_L1normalized))
#print(Train_Labels)
#prediction for training L2 normalized data using SCIKIT toolkit
MSE_trainl2=np.mean((logSKLl2.predict(Train_L2normalized)-Train_Labels)**2)
print MSE_trainl2
#print(logSKLl2.predict(Train_L2normalized))
#print(Train_Labels)



#prediction for test data using SCIKIT toolkit
MSE_test=np.mean((logSKL.predict(Test_Feature)-Test_Labels)**2)
print MSE_test
#print(logSKL.predict(Test_Feature))
#print(Test_Labels)
#prediction for test L1 normalized data using SCIKIT toolkit
MSE_testl1=np.mean((logSKLl1.predict(Test_L1normalized)-Test_Labels)**2)
print MSE_testl1
#print(logSKLl1.predict(Test_L1normalized))
#print(Test_Labels)
#prediction for test L2 normalized data using SCIKIT toolkit
MSE_testl2=np.mean((logSKLl2.predict(Test_L2normalized)-Test_Labels)**2)
print MSE_testl2
#print(logSKLl2.predict(Test_L2normalized))
#print(Test_Labels)




		




