import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import math


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
data = mnist.data[:100]  
labels = mnist.target.astype(int)

class CustomMeanShift:
    def __init__(self, bandwidth, max_iters=50):
        self.bandwidth = bandwidth
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, data):
        centroids = data.copy()
        tree = KDTree(data)  
        iteration = 0

        while iteration < self.max_iters:
            new_centroids = []

            for point in centroids:
                indices = tree.query_radius([point], r=self.bandwidth)[0]
                in_bandwidth = data[indices]  

                
                new_centroid = np.mean(in_bandwidth, axis=0)
                new_centroids.append(new_centroid)

            new_centroids = np.unique(new_centroids, axis=0)

            
            if len(new_centroids) == len(centroids) and np.allclose(new_centroids, centroids, atol=1e-3):
                break

            centroids = new_centroids
            iteration += 1

            
            if iteration % 1 == 0 or iteration == self.max_iters - 1:
                self.plot_centroids(centroids, iteration)

        self.centroids = centroids

    def plot_centroids(self, centroids, iteration):
        num_centroids = len(centroids)
        cols = min(5, num_centroids)  
        rows = math.ceil(num_centroids / cols)  

        plt.figure(figsize=(15, rows * 3))  

        for i, centroid in enumerate(centroids):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(centroid.reshape(28, 28), cmap='gray')
            plt.axis('off')
            plt.title(f"Cluster {i + 1} - Iter {iteration}")

        plt.tight_layout()
        plt.show()


bandwidth = 2000
model = CustomMeanShift(bandwidth=bandwidth)
model.fit(data)
plt.show()