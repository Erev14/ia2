from SVM import SVM
from plotting import *
import numpy as np

np.random.seed(24)

mean1 = np.array([0, 2])
mean2 = np.array([2, 0])
cov = np.array([[1.5, 1], [1, 1.5]])
X1 = np.random.multivariate_normal(mean1, cov, 100)
X2 = np.random.multivariate_normal(mean2, cov, 100)

x = np.vstack((X1, X2))
y = np.hstack((np.ones(len(X1)), np.ones(len(X1)) * -1))

# problem 2
# from sklearn.datasets import make_moons
# x, y = make_moons(200, noise = 0.2)
# y = np.asanyarray(y, dtype=np.float64)
# y += (y == 0) * - 1.0

# solution
model = SVM(kernel='gaussian', C=100, gamma=0.1)
model.fit(x, y)
plot_svm(x, y, model)