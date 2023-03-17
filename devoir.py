import numpy as np

class KMeans:
    def __init__(self, k=2, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize cluster centers randomly
        self.centers = np.random.randn(self.k, n_features)

        for i in range(self.max_iter):
            # Assign each sample to the closest cluster center
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers, axis=2)
            cluster_labels = np.argmin(distances, axis=1)

            # Update cluster centers to be the mean of the samples assigned to them
            for j in range(self.k):
                self.centers[j] = np.mean(X[cluster_labels == j], axis=0)

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers, axis=2)
        return np.argmin(distances, axis=1)
