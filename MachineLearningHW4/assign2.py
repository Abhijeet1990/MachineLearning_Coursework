import os,sys
import numpy as np
from string import split
from sklearn import linear_model
from sklearn import preprocessing
from numpy.linalg import inv

class LogisticRegressionSelf:

    def probability(self):
      #print(self.w)
      product=self.features.dot(self.w)
      expr=np.exp(-product)
      return 1/(1+expr)


    #this function calculates w for each test data
    def weight_calc(self,test_data,tau=100):
      #print(tau)
      for i in range(self.features.shape[0]):
      	#self.weight[i]= np.exp(-(np.sum(np.square(test_data.T-self.features[i,1:]))/self.features.shape[0])/(2*np.square(tau)))
        self.weight[i]= np.exp(-(np.sum(np.square(test_data.T-self.features[i,1:])))/(2*np.square(tau)))
      return self.weight
      
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
      #print("in grad")
      error=np.zeros((self.features.shape[0],1))
      p=self.probability()
      for i in range(self.features.shape[0]):
         error[i] = self.weight[i]*(self.labels[i]-p[i]) 
      #print error
      #error=self.weight*(self.labels-self.probability())
      product = error*self.features
      #print(error.shape)
      #print(self.features.shape)
      #print(product.shape)
      grad=product.sum(axis=0).reshape(self.w.shape)
      grad -= 0.001*self.w
      #print(grad)
      return grad


    def Hessian(self):
      p=self.probability()
      #print("hessian")
      #print(p.shape)
      #print(self.weight)
      
      #print(self.weight)
      #print(p)
      #print(1-p)
      Dii=np.zeros(self.features.shape[0])
      for i in range(self.features.shape[0]):
      	Dii[i]= self.weight[i]*p[i]*(1-p[i]) #there is a bug in iteration no. 2
      #print(Dii)
      D_i_i =  np.reshape (Dii,self.features.shape[0])
      #print(D_i_i.shape)
      m=np.diag(D_i_i)
      #print(m)
      hess=np.mat(self.features.T)*np.mat(m)*np.mat(self.features) 
      hess -= 0.001*np.identity(self.features.shape[1])
      #print(hess)
      return hess

    def newton_method(self,max_iter=5):
       iteration=0
       prev=self.Hessian()
       while(iteration < max_iter):
	  self.w = self.w - inv(prev)*self.log_likelihood_gradient()
          #print(self.w)
          temp=self.Hessian()
	  prev=temp
          iteration+=1
     

    def __init__(self,X,y):
        
        #self.tolerance = tolerance
        self.labels = y.reshape(y.size,1)
        self.w = np.zeros((X.shape[1]+1,1))
        self.features = np.ones((X.shape[0],X.shape[1]+1))
        self.features[:,1:] = X
	self.weight = np.zeros((X.shape[0],1));


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

log = LogisticRegressionSelf(Train_Feature,Train_Labels)
likelihood = log.weighted_log_likelihood()
#print("Likelihood func")
#print(likelihood)

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
#print(Test_Feature)
Tau=np.array([5.0,1.0,0.5,0.1,0.05,0.01])
prob1=np.zeros(len(Test_Labels))
for j in range(len(Tau)): 
  for i in range(len(Test_Labels)):
    w_i=log.weight_calc(Test_Feature[i,:],Tau[j])
    
    log.newton_method()
    #print(log.w.shape)
    product_test=Test_Feature.dot(log.w[1:,:])
    #print(log.w[1:,:])
    #print(product_test.shape)
    prob1[i]= 1/(1+np.exp(-product_test[i]))
    
    if prob1[i] > 0.5:
	prob1[i]=1.0
    else:
 	prob1[i]=0.0


  MSE_test1=np.mean((prob1-Test_Labels)**2)
  print "Testing error"
  print MSE_test1
  print "error%"
  print np.sum(np.absolute(prob1-Test_Labels))/len(Test_Labels)*100





		




