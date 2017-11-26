import numpy as np
from operator import itemgetter
A = np.array([1,2,3,4,5,6])
#B = vec2matrix(A,ncol=2)
#B = np.reshape(A, (-1, 2))
A.reshape((-1,2))
print A

lst = [
   ['John' '2'],
 ['Jim' '9'],
['Jason' '1']
 ]
print lst
lst.sort(key=itemgetter(0))
print lst





