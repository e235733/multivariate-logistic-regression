from sklearn.datasets import load_iris
import numpy as np

from logistic_regression import LogisticRegressionModel as lrm
from plotter import Plotter

INTERVAL = 0.05
NUM_STEP = 300

ETA_W = 0.0005
ETA_B = 0.02

iris = load_iris()

X = iris.data
Y = iris.target

models = []
for c in np.unique(Y):
    # One-vs-Rest
    Y_ovr = np.where(Y == c, 1, 0)
    
    model_c = lrm(X, Y_ovr, 3, ETA_W, ETA_B)
    models.append(model_c)

plt = Plotter(INTERVAL, X, Y)

for i in range(NUM_STEP):
    for model in models:
        model.shift()
    
    plt.show(models, i+1)

# 正解率の計算
probs = []
for model in models:
    z = X @ model.w + model.b
    p = 1 / (1 + np.exp(-z))
    probs.append(p)

probs = np.array(probs)

Y_pred = np.argmax(probs, axis=0)

accuracy = np.mean(Y_pred == Y)

print(f"Accuracy: {accuracy * 100:.2f}%")
