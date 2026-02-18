from sklearn.datasets import make_blobs

from logistic_regression import LogisticRegressionModel as lrm
from plotter import Plotter

N_SAMPLES = 100
N_FEATURES = 2
N_CENTERS = 3
CLUSTER_STD = 10
CLUSTER_RANGE = 100

INTERVAL = 0.05
NUM_STEP = 100

ETA_W = 0.01
ETA_B = 0.01


X, Y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CENTERS, cluster_std=CLUSTER_STD, center_box=(-CLUSTER_RANGE, CLUSTER_RANGE))
print(X, Y)

plt = Plotter(INTERVAL, X, Y)

model = lrm(X, Y, ETA_W, ETA_B)
for i in range(NUM_STEP):
    model.shift()
    plt.show(model, i+1)
