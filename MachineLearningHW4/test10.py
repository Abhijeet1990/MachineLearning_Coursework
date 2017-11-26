


import numpy as np


a=np.array([2,3,4,5])
b=np.array([3,4,5,6])
print a*b.T
print np.dot(a,b)
print np.sum(a*b.T)

c=np.array([[1],[2],[3],[4]])
print (c-b)**2
MSE=np.mean((b-c)**2)
print MSE
