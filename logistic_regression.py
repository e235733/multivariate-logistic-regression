import numpy as np

class LogisticRegressionModel:
    def __init__(self,explain,depend,features,eta):
        #説明変数X(d次元列ベクトルn個分), 目的変数y(n個分の0か1のラベル)
        self.X = explain
        self.N = self.X.shape[0]
        # Y は one_hot 表現にする
        self.Y = np.identity(features)[depend]
        #調整すべきパラメータb:切片、w:d次元分の傾きを作成
        self.dim = self.X.shape[1]
        self.W = np.zeros((self.dim, features)) 
        self.b = np.zeros(features)
        #bとwの学習率
        self.eta = eta

        self.loss_history = []

    def calc_loss(self):
        #損失評価
        logP = np.log(self.P)
        loss = -np.sum(self.Y * logP) / self.N
        print("loss: ", loss)
        self.loss_history.append(loss)

    def calc_P(self):
        Z = self.X @ self.W + self.b
        Z_max = np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z - Z_max)
        self.P = exp_Z / np.sum(exp_Z,axis=1,keepdims=True)

    def grad(self):
        # 予測値
        #print(np.shape(self.X), np.shape(self.W))
        self.calc_P()
        dz = self.P - self.Y
        # 勾配
        grad_w = self.X.T @ dz / self.N
        grad_b = np.sum(dz,axis=0) / self.N

        return grad_w, grad_b
    
    def shift(self):
        # 勾配を計算
        grad_w, grad_b = self.grad()
        #print(f"grad_w: {grad_w}, grad_b: {grad_b:.4f}")

        # パラメータの更新
        self.W -= self.eta * grad_w
        self.b -= self.eta * grad_b

if __name__ == "__main__":
    # サンプルデータ
    rng = np.random.default_rng()
    num_data = 50 # 各クラス50個

    # クラス0, 中心が x=-2, Y=-2
    X_0 = rng.normal(loc=-2.0, scale=1.0, size=(2, num_data))
    y_0 = np.zeros(num_data, int)

    # クラス1, 中心が x=2, Y=2
    X_1 = rng.normal(loc=2.0, scale=1.0, size=(2, num_data))
    y_1 = np.ones(num_data, int)

    explain = np.hstack((X_0, X_1)).T
    depend = np.concatenate((y_0, y_1))
    indices = np.arange(100)
    rng.shuffle(indices)
    explain = explain[indices]
    depend = depend[indices]

    model = LogisticRegressionModel(explain, depend, 2, 0.01, 0.01)

    print(model.grad())