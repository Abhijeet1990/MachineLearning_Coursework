import numpy as np
import numpy.matlib


a = np.arange(25).reshape(5,5)
b = np.arange(5)
c = np.arange(6).reshape(2,3)

print a
print b
print c
print np.einsum('ii', a)

print np.einsum(a, [0,0])

print np.einsum('ii->i', a)

print np.einsum(a, [0,0], [0])

print np.einsum('ij,j', a, b)

print np.einsum(a, [0,1], b, [1])

print np.einsum('ji', c)

print np.einsum(c, [1,0])
print a*a
print np.einsum('ij,ij->i', a,a)

print np.matlib.repmat(b,5,1)

for k in range(10):
   print k

m = np.arange(5)
X = lambda a,b: a+b

print X(b,m)



