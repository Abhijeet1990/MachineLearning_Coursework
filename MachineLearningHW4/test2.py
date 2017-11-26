import numpy as np
from numpy.linalg import inv
a=np.array([[1,2],[3,4],[6,7]])
b=np.array((5,5,5,7))
c=np.reshape(b, (1,4))
l=np.array((3,4,5,6))
d=np.reshape(l, (1,4))
print(c.shape)
print(d.shape)
print(c*d)

