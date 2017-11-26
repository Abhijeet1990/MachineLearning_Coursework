import numpy as np
#pylab inline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.mlab import PCA as mlabPCA

np.random.seed(4294967294) # random seed for consistency

# A reader pointed out that Python 2.7 would raise a
# "ValueError: object of too small depth for desired array".
# This can be avoided by choosing a smaller random seed, e.g. 1
# or by completely omitting this line, since I just used the random seed for
# consistency.

mu_vec1 = np.array([0,0,0,0])
cov_mat1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (4,20), "The matrix has not the dimensions 3x20"

print(class1_sample)

mu_vec2 = np.array([1,1,1,1])
cov_mat2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class1_sample.shape == (4,20), "The matrix has not the dimensions 3x20"

print(class2_sample)

all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (4,40), "The matrix has not the dimensions 3x40"

print(all_samples.T)

mlab_pca = mlabPCA(all_samples.T)

print(mlab_pca.Y)
print('PC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)

plt.plot(mlab_pca.Y[0:20,0],mlab_pca.Y[0:20,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(mlab_pca.Y[20:40,0], mlab_pca.Y[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.legend()
plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

plt.show()
