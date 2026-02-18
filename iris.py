from sklearn.datasets import load_iris
import numpy as np

from logistic_regression import LogisticRegressionModel as lrm
from plotter import Plotter

INTERVAL = 0.05
NUM_STEP = 300

ETA_W = 0.1
ETA_B = 1

iris = load_iris()

X = iris.data
Y = iris.target

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