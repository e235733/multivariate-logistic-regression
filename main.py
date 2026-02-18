from sklearn.datasets import make_blobs
import numpy as np

from logistic_regression import LogisticRegressionModel as lrm
from plotter import Plotter

N_SAMPLES = 100
N_FEATURES = 2
N_CENTERS = 3
CLUSTER_STD = 10
CLUSTER_RANGE = 100

INTERVAL = 0.05
NUM_STEP = 100

ETA_W = 0.00001
ETA_B = 0.01

X, Y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CENTERS, 
                  cluster_std=CLUSTER_STD, center_box=(-CLUSTER_RANGE, CLUSTER_RANGE))

models = []
for c in np.unique(Y):
    # One-vs-Rest
    Y_ovr = np.where(Y == c, 1, 0)
    
    model_c = lrm(X, Y_ovr, N_FEATURES, ETA_W, ETA_B)
    models.append(model_c)

plt = Plotter(INTERVAL, X, Y)

for i in range(NUM_STEP):
    for model in models:
        model.shift()
    
    plt.show(models, i+1)