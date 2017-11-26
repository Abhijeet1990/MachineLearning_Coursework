import numpy as np
from sklearn.decomposition import PCA

X = np.array([[0, 1,0,0], [0, 0,0,0], [0, 0,1,1], [1, 0,0,0], [0, 1,0,0], [0, 0,1,0]])
pca = PCA(n_components=3)
pca.fit(X)
y=pca.transform(X)
#PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,svd_solver='auto', tol=0.0, whiten=False)
print(pca.explained_variance_ratio_)
print(y)
