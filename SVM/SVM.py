import numpy as np
import cvxopt

class SVM():

    def __init__(self, kernel='linear', C = 0.001, gamma = 0.001, degree=3):
        self.C = float(C)
        self.gamma = float(gamma)
        self.d = int(degree)
        self.kernels = {
            'linear': self.linear,
            'poly': self.poly,
            'gaussian': self.gaussian
        }
        self.kernel = self.kernels[kernel]

    def linear(self, xi, xj):
        return np.dot(xi, xj)

    def poly(self, xi, xj):
        return (np.dot(xi,xj) + 1) ** self.d

    def gaussian(self, xi, xj):
        return np.exp(-self.gamma * np.linalg.norm(xi - xj)**2)
    

    def fit(self, X, y):
        # calc in i size
        n_samples, n_features = X.shape

        # cal Gramm matiz with the kernel
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        # Resolve SVM  con cvxopt package
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C == 0:
            G = cvxopt.matrix(-1 * np.identity(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            temp1 = -1 * np.identity(n_samples)
            temp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((temp1, temp2)))
            temp1 = np.zeros(n_samples)
            temp2 = self.C * np.ones(n_samples)
            h = cvxopt.matrix(np.hstack((temp1, temp2)))
        # disable outputs
        cvxopt.solvers.options['show_progress'] = False
        
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        # extract lagrange multipliers
        lamb = np.ravel(solution['x'])

        # detect soport vectors
        mask = lamb > 1e-5
        ind = np.arange(len(lamb))[mask]
        self.lamb = lamb[mask]

        # extact soport vectors
        self.sv = X[mask]
        self.sv_y = y[mask]

        # calculate b sesg
        self.b = 0
        for i in range(len(self.lamb)):
            self.b += self.sv_y[i]
            self.b -= np.sum(self.lamb[i] * self.sv_y[i] * K[ind[i], mask])
        self.b = self.b / len(self.lamb)

    def project(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.lamb, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_pred[i] = s
        return y_pred + self.b

    def predict(self, X):
        return np.sign(self.project(X))
