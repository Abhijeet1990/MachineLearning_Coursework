import numpy as np
#pylab inline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.mlab import PCA as mlabPCA

arr=np.array([[0,0,1,2,2,0,0],[0,0,4,5,6,0,0],[1,2,0,0,0,0,7]])
#assume each point is seven dimensional and we have 3 points   
print (arr.shape)
mlab_pca = mlabPCA(arr.T)
print(mlab_pca.Y)
