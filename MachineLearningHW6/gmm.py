

import numpy as np
import pylab as plt
from collections import namedtuple
import os,sys
import numpy.matlib
from sklearn import preprocessing

#the problem with this code is that the data given has a feature of column with all zeros....So when the Sigma is upgrated in the 
#second iteration the determinant of the sigma becomes zero and the calculation terminates with a singular matrix..
#How to resolve this?????

nComp= 10 #no of gaussian component
thres= 0.0001 #threshold to stop the EM iteration
params = namedtuple('params', ['mu', 'Sigma', 'w', 'log_likelihoods', 'num_iters'])

def P(X,mu,Sigma):
    print np.linalg.det(Sigma)
    
    M = np.linalg.det(Sigma)**-0.5**(2 *np.pi)**(-X.shape[1]/2.)
    print "M is",M
    mu_new = np.matlib.repmat(mu,X.shape[0],1)
    #print mu_new
    N = np.exp(-0.5*np.einsum('ij, ij -> i',X - mu_new, np.dot(np.linalg.inv(Sigma) , (X - mu_new).T).T ) )
    print "N is",N
    #print M*N
    return M*N

def fit_EM (X, max_iters = 1000):
          
    n, d = X.shape # n = number of data-points, d = no. of features for each data point
    # randomly choose the starting centroids/means   
    #print n
    #print d     
    mu = X[np.random.choice(n, nComp, False), :]  
    # initialize the covariance matrices for each gaussians
    #print mu
    Sigma= [np.eye(d)] * nComp
    # initialize the probabilities/weights for each gaussians
    #print Sigma
    #print np.linalg.inv(Sigma)
    w = [1./nComp] * nComp
    # responsibility matrix is initialized to all zeros
    # we have responsibility for each of n points for eack of k gaussians
    R = np.zeros((n, nComp))
    # log_likelihoods
    log_likelihoods = []
    '''
    P= np.linalg.det(Sigma) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(Sigma) , (X - mu).T).T ) )
    
    P = lambda mu, S: np.linalg.det(S) ** -.5 ** (2 * np.pi) ** (-X.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        X - mu, np.dot(np.linalg.inv(S) , (X - mu).T).T ) ) 
    '''
    #print P

    # Iterate till max_iters iterations        
    while len(log_likelihoods) < max_iters:
            
         # E - Step
            
         ## Vectorized implementation of e-step equation to calculate the 
         ## membership for each of k -gaussians
         for k in range(nComp):
            #print "k",k
            print "mu ",mu[k]
            #print "sigma",Sigma[k]
            R[:, k] = w[k] * P(X,mu[k], Sigma[k])
         print "Responsibility Matrix",R
         ### Likelihood computation
         log_likelihood = np.sum(np.log(np.sum(R, axis = 1)))
            
         log_likelihoods.append(log_likelihood)
            
         ## Normalize so that the responsibility matrix is row stochastic
         R = (R.T / np.sum(R, axis = 1)).T
         print "Responsibility Matrix after normalization",R 
         ## The number of datapoints belonging to each gaussian            
         N_ks = np.sum(R, axis = 0)
         print "number of data belonging to each gaussian",N_ks

         # M Step
         ## calculate the new mean and covariance for each gaussian by 
         ## utilizing the new responsibilities
         for k in range(nComp):
                
             ## means
             mu[k] = 1. / N_ks[k] * np.sum(R[:, k] * X.T, axis = 1).T
             print mu[k]
             print X[:,2]
             x_mu = np.matrix(X - mu[k])
             print "x_mu",x_mu.shape
             ## covariances
             Sigma[k] = np.array(1 / N_ks[k] * np.dot(np.multiply(x_mu.T,  R[:, k]), x_mu))
             print np.linalg.det(Sigma[k])   
             ## and finally the probabilities
             w[k] = 1. / n * N_ks[k]

         # check for convergence
         # if len(log_likelihoods) < 2 : continue
         # if np.abs(log_likelihood - log_likelihoods[-2]) < thres : break

    ## bind all results together
    
    
    params.mu = mu
    params.Sigma = Sigma
    params.w = w
    params.log_likelihoods = log_likelihoods
    params.num_iters = len(log_likelihoods)  
   
    return params 


def plot_log_likelihood(X):
    
    plt.plot(X)
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()

def predict(x,params):
    p = lambda mu, s : np.linalg.det(s) ** - 0.5 * (2 * np.pi) **(-len(x)/2) * np.exp( -0.5 * np.dot(x - mu ,np.dot(np.linalg.inv(s) , x - mu)))
    probs = np.array([w * p(mu, s) for mu, s, w in zip(params.mu, params.Sigma, params.w)])
    return probs/np.sum(probs)


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

print Train_Feature
X_train = Train_Feature
y_train = Train_Labels
X_test = Test_Feature
y_test = Test_Labels

max_iters=1
params = fit_EM(X_train, max_iters)
print params.log_likelihoods
plot_log_likelihood(params.log_likelihoods)
print predict(X_test,params)



