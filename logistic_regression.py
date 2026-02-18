import numpy as np

class LogisticRegressionModel:
    def __init__(self,explain,depend,features,eta_w,eta_b):
        #説明変数X(d次元列ベクトルn個分), 目的変数y(n個分の0か1のラベル)
        self.X = explain
        # Y は one_hot 表現にする
        self.dim = self.X.shape[1]
        self.Y = np.identity(self.dim)[depend]
        #調整すべきパラメータb:切片、w:d次元分の傾きを作成
        self.W = np.zeros((self.dim, features)) 
        self.b = np.zeros(features)
        #bとwの学習率
        self.eta_w = eta_w
        self.eta_b = eta_b

    def grad(self):
        # 予測値
        print(np.shape(self.X), np.shape(self.W))
        Z = self.X @ self.W + self.b
        P = np.exp(Z) / np.sum(np.exp(Z))
        print(Z, P)

        # 誤差
        d = self.Y - P
        # 勾配
        grad_w = d @ self.X
        grad_b = np.sum(d)

        return grad_w, grad_b 
    
    def shift(self):
        # 勾配を計算
        grad_w, grad_b = self.grad()
        #print(f"grad_w: {grad_w}, grad_b: {grad_b:.4f}")

        # パラメータの更新
        self.W += self.eta_w * grad_w
        self.b += self.eta_b * grad_b

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

    model = LogisticRegressionModel(explain, depend, 3, 0.01, 0.01)
    print(model.W)
    model.grad()