import numpy as np
a=np.array([1,2,3])
b=np.array([2,3,4])
a=np.reshape(a,(3,1))
b=np.reshape(b,(3,1))
print np.dot(a,b.T)
print a*b.T
