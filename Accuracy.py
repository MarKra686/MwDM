import numpy as np
from Mean_Shift_MNIST import CustomMeanShift
from sklearn.datasets import fetch_openml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def assign_labels(X, centroids):
    labels = []
    for point in X:
        distances = np.linalg.norm(centroids - point, axis=1)
        labels.append(np.argmin(distances))
    return np.array(labels)

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data[:100]  
y_true = mnist.target[:100].astype(int)  


model = CustomMeanShift(bandwidth=50, max_iters=10)
model.fit(X)

predicted_labels = assign_labels(X, model.centroids)

ari = adjusted_rand_score(y_true, predicted_labels)
nmi = normalized_mutual_info_score(y_true, predicted_labels)

print("Adjusted Rand Index (ARI):", ari)
print("Normalized Mutual Information (NMI):", nmi)
