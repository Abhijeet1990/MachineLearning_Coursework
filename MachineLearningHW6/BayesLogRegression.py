'''
function [w,Smat,alpha,loglik]=BayesLogRegression(phi,c,w,alpha,R,opts)
%[w,Smat,alpha,loglik]=BayesLogRegression(phi,c,w,alpha,R,opts)
% Bayesian Logistic Regression
%
% Inputs:
% phi : M*N matrix of phi vectors on the N training points
% c : N*1 vector of associated class lables (0,1)
% w : initial weight vector
% alpha : initial regularisation parameter
% opts.HypUpdate : 1 for EM, 2 for Gull-MacKay
% opts.HypIterations : number of hyper parameter updates
% opts.NewtonIterations : number of Newton Updates
%
% Outputs:
% w : learned posterior mean weight vector
% Smat : posterior covariance
% alpha : learned regularisation parameter
% loglik : log likelihood of training data for optimal parameters
import brml.*
s=2*c(:)-1; [M N]=size(phi); logdetR = logdet(R); alphas=[];
for alphaloop=1:opts.HypIterations
    for wloop=1:opts.NewtonIterations % Newton update for Laplace approximation
        sigmawh = sigma(s.*(phi'*w));
        gE=alpha.*R*w; J=zeros(M);
        for n=1:N
            gE = gE-(1-sigmawh(n))*phi(:,n).*s(n);
            J = J + sigmawh(n)*(1-sigmawh(n))*phi(:,n)*phi(:,n)';
        end
        Hess= alpha*R+ J;
        w = w-0.5*(Hess\gE);
    end
    Smat = inv(Hess);
    L(alphaloop)=-0.5*alpha*w'*R*w+sum(logsigma(s.*(phi'*w)))-0.5*logdet(Hess)+M*0.5*log(alpha)+0.5*logdetR;
    switch opts.HypUpdate
        case 1
            alpha = M./(w'*R*w+trace(R*Smat)); % EM update
        case 2
            alpha = min(10000,(M-alpha.*trace(R*Smat))./(w'*R*w)); % MacKay/Gull update
    end
    alphas=[alphas alpha];
    if opts.plotprogess
        subplot(1,3,1); plot(L); title('likelihood');
        subplot(1,3,2); plot(log(alphas)); title('log alpha');
        subplot(1,3,3); bar(w); title('mean weights');drawnow
    end
end
loglik=L(end);


s=1./(1+exp(-x))
'''
import numpy as np
from sklearn.utils.extmath import randomized_svd
from numpy.linalg import inv
import math
import os,sys

def BayesLogRegression(phi,c,w,alpha,R,HypIterations,NewtonIterations):
   s=2*c-1
   s=np.reshape(s,(phi.shape[0],1))
   M=phi.shape[0]
   N=phi.shape[1]
   J=np.zeros((M,N)) 
   U, Sigma, VT = randomized_svd(R,n_components=15,n_iter=5,random_state=None)
   logdetR = np.sum(np.log(np.diag(Sigma)+1.0e-20))
   alphas = []
   #L=np.zeros(HypIterations)
   w=np.reshape(w,(phi.shape[0],1))
   print R.shape
   for alphaloop in range(HypIterations):
        for wloop in range(NewtonIterations):
             g=phi.T*w
             y=np.exp(-np.dot(s,g))
             print"y shape",y.shape
             sigmawh = 1/(1+y)
             gE=np.dot(alpha,np.mat(R)*np.mat(w))
             print "gE shape",gE.shape
             print "sigmawh shape",sigmawh.shape
             for n in range(N):
                 x=np.dot((1-sigmawh[n])*phi[:,n],s[n])
                 x=np.reshape(x,(phi.shape[0],1))
                 gE = gE-x
                 J = J + sigmawh[n]*(1-sigmawh[n])*phi[:,n]*phi[:,n].T
             Hess= alpha*R+ J
             w = w-0.5*np.mat(inv(Hess))*np.mat(gE)
             
        Smat=inv(Hess)
        '''
        expr1 = -0.5*alpha*w*R*w.T
        print "expr1",expr1
        expr2 = np.sum(-np.log(1+np.exp(-np.dot(s,(phi.T*w)))))
        print "expr2",expr2
        U, Sigma, VT = randomized_svd(Hess,n_components=15,n_iter=5,random_state=None)
        expr3 = -0.5*np.sum(np.log(np.diag(Sigma)+1.0e-20))
        print "expr3",expr3
        expr4 = M*0.5*np.log(alpha)
        print "expr4",expr4
        expr5 = 0.5*logdetR
        print "expr5",expr5
        print((expr1+expr2+expr3+expr4+expr5).shape)
        L[alphaloop] = expr1+expr2+expr3+expr4+expr5
        '''
        alpha = np.dot(M,1/(w.T*R*w + np.trace(R*Smat)))
                
   return (w,Smat,alpha)

def kernel(x,xp,lamda,noise):
   delx = x-xp
   delxm = np.reshape(delx,(delx.shape[0],1))
   #print delxm.shape
   k = np.exp(-lamda*(np.mat(delxm.T)*np.mat(delxm)))
   #print k.shape
   k = k + noise*np.identity(k.shape[0])
   #print k
   return k

def avisigmaGauss(mn,v):
   print mn.shape
   print v.shape
   erflambda=np.sqrt(3.14)/4
   expr1=np.power(erflambda,2*v)
   print expr1.shape
   out = 0.5 + 0.5*math.erf(erflambda*mn/np.sqrt(1+2*(expr1)))
   return out

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
phi=np.ones((Train_Feature.shape[0],Train_Feature.shape[0]))

for i in range(Train_Feature.shape[0]):
    for j in range(Train_Feature.shape[0]):
        phi[i][j] = kernel(Train_Feature[j,:],Train_Feature[i,:],2,0.00001)
print phi

w,Smat,alpha=BayesLogRegression(phi,Train_Labels,np.zeros(Train_Feature.shape[0]),1,np.identity(Train_Feature.shape[0]),150,10)
print "w shape",w.shape

for n in range(Train_Feature.shape[0]):
   mn=np.mat(w.T)*np.mat(phi[n,:].T)
   print mn
   v=phi[n,:]*Smat*phi[n,:].T
   print v
   ptrain[n] = avisigmaGauss(mn,v)
   if ptrain[n] > 0.5:
	ptrain[n]=1.0
   else:
 	ptrain[n]=0.0

print "Training error%"
print np.sum(np.absolute(ptrain-Train_Labels))/len(Train_Labels)*100

phi2=np.ones((Test_Feature.shape[0],Test_Feature.shape[0]))
for i in range(Train_Feature.shape[0]):
    for j in range(Test_Feature.shape[0]):
        phi2[i][j] = kernel(Test_Feature[j,:],Train_Feature[i,:],2,0.00001)


for n in range(Test_Feature.shape[0]):
   ptest[n] = avisigmaGauss(w.T*phi2[n,:],phi2[n,:].T*Smat*phi2[n,:])
   if ptest[n] > 0.5:
	ptest[n]=1.0
   else:
 	ptest[n]=0.0

print "Testing error%"
print np.sum(np.absolute(ptest-Test_Labels))/len(Test_Labels)*100










