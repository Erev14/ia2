import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

class RBFNN:
    def __init__(self, n_hidden) -> None:
        self.nh = n_hidden
    
    def predict(self, X):
        X = np.asanyarray(X)
        print(self.C.shape)
        print(X.shape)
        G = np.exp(-(distance.cdist(X, self.C))**2 / self.sigma**2)
        return G @ self.W
    
    def predict_class(self, X):
        G = np.exp(-(distance.cdist(X, self.C))**2 / self.sigma**2)
        return np.argmax(G @ self.W, axis=1)
    
    def fit(self, X, Y):
        self.ni, self.no = X.shape[1], Y.shape[1]
        km = KMeans(n_clusters=self.nh).fit(X)
        self.C = km.cluster_centers_
        self.sigma = (self.C.max() - self.C.min()) / np.sqrt(2 * self.nh)
        G = np.exp(-(distance.cdist(X, self.C))**2 / self.sigma**2)
        self.W = np.linalg.pinv(G) @ Y