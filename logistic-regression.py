import numpy as np

class LogisticRegressionModel:
    def __init__(self,explain,depend,eta_w,eta_b):
        #説明変数X(d次元列ベクトルn個分), 目的変数y(n個分の0か1のラベル)
        self.X = explain
        self.y = depend
        #調整すべきパラメータb:切片、w:d次元分の傾きを作成
        self.num_features = self.X.shape[0]
        self.w = np.zeros(self.num_features)
        self.b = 0.0
        #bとwの学習率
        self.eta_w = eta_w
        self.eta_b = eta_b

    def grad(self):
        # 予測値
        z = self.w @ self.X + self.b
        p = 1 / (1 + np.exp(-z))

        # 誤差
        d = self.y - p

        # 勾配
        grad_w = d @ self.X.T
        grad_b = np.sum(d)

        return grad_w, grad_b 
    
    def shift(self):
        # 勾配を計算
        grad_w, grad_b = self.grad()
        print(f"grad_w: {grad_w}, grad_b: {grad_b:.4f}")

        # パラメータの更新
        self.w += self.eta_w * grad_w
        self.b += self.eta_b * grad_b

if __name__ == "__main__":
    # サンプルデータ
    rng = np.random.default_rng()
    num_data = 50 # 各クラス50個

    # クラス0, 中心が x=-2, y=-2
    X_0 = rng.normal(loc=-2.0, scale=1.0, size=(2, num_data))
    y_0 = np.zeros(num_data)

    # クラス1, 中心が x=2, y=2
    X_1 = rng.normal(loc=2.0, scale=1.0, size=(2, num_data))
    y_1 = np.ones(num_data)

    explain = np.hstack((X_0, X_1))
    depend = np.concatenate((y_0, y_1))
    indices = np.arange(100)
    rng.shuffle(indices)
    explain = explain[:, indices]
    depend = depend[indices]

    # 学習
    model = LogisticRegressionModel(explain, depend, 0.01, 0.01)

    for i in range(20):
        model.shift()