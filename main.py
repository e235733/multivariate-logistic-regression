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

ETA_W = 0.002
ETA_B = 2

X, Y = make_blobs(n_samples=N_SAMPLES, n_features=N_FEATURES, centers=N_CENTERS, 
                  cluster_std=CLUSTER_STD, center_box=(-CLUSTER_RANGE, CLUSTER_RANGE), random_state=None)

model = lrm(X, Y, 3, ETA_W, ETA_B)

plt = Plotter(INTERVAL, X, Y)
model.calc_P()
model.calc_loss()
plt.show(model, 0)

for i in range(NUM_STEP):
    model.shift()
    model.calc_loss()
    
    plt.show(model, i+1)

# 正解率の計算
Y_pred = np.argmax(model.P, axis=1)
accuracy = np.mean(Y_pred == Y)
print(f"Accuracy: {accuracy * 100:.2f}%")