import numpy as np
from sklearn.datasets import make_blobs

def mean_shift(data, bandwidth, max_iter=300, tol=1e-3):

    centroids = data.copy()
    n_samples, n_features = data.shape

    for it in range(max_iter):
        new_centroids = []
        for i in range(n_samples):
            point = centroids[i]

            distances = np.linalg.norm(data - point, axis=1)
            neighbors = data[distances < bandwidth]

            new_centroid = neighbors.mean(axis=0)
            new_centroids.append(new_centroid)

        new_centroids = np.array(new_centroids)

        shifts = np.linalg.norm(new_centroids - centroids, axis=1)
        if np.all(shifts < tol):
            break

        centroids = new_centroids

    unique_centroids = np.unique(np.round(centroids, decimals=3), axis=0)
    labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - unique_centroids, axis=2), axis=1)

    return unique_centroids, labels

data, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

bandwidth = 2.5
centroids, labels = mean_shift(data, bandwidth)
print("Centroids: ", centroids)
'print("Labels: ", labels)'