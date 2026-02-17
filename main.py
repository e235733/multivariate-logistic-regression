from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

N_SAMPLES = 100
N_FEATURES = 2
N_CENTERS = 3
CLUSTER_STD = 10
CLUSTER_RANGE = 100


X, Y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CENTERS, cluster_std=CLUSTER_STD, center_box=(-CLUSTER_RANGE, CLUSTER_RANGE))
print(X, Y)

plt.figure(figsize=(6, 6))
plt.title("generated data")
plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()