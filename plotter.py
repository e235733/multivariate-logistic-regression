import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, interval, explain, depend):
        self.interval = interval
        self.explain = explain
        self.depend = depend
        
        # データの次元数 を取得
        self.dim = explain.shape[1] 
        print(np.shape(explain))
        print(self.dim)

        # 学習曲線の履歴
        self.loss_history = []
        
        self.fig = plt.figure(figsize=(12, 6)) 
        
        # 次元数に応じて左側のグラフの種類を変える
        if self.dim == 1:
            self.ax_data = self.fig.add_subplot(1, 2, 1)
        elif self.dim == 2:
            self.ax_data = self.fig.add_subplot(1, 2, 1)
        elif self.dim == 3:
            self.ax_data = self.fig.add_subplot(1, 2, 1, projection='3d')
        else:
            # 4次元以上の場合は ax_data を作らない
            self.ax_data = None 
            
        # 右側のグラフは学習曲線用
        if self.dim <= 3:
            self.ax_loss = self.fig.add_subplot(1, 2, 2)
        # 4次元以上の場合は、ウィンドウ全体を学習曲線にする
        else:
            self.ax_loss = self.fig.add_subplot(1, 1, 1)

    # 交差エントロピー誤差
    def _calc_loss(self, model):
        z = model.w @ self.explain.T + model.b
        p = 1 / (1 + np.exp(-z))
        epsilon = 1e-15 # log(0)のエラーを防ぐ微小値
        loss = -np.mean(self.depend * np.log(p + epsilon) + (1 - self.depend) * np.log(1 - p + epsilon))
        return loss

    # メインの描画メソッド
    def show(self, model, step):
        # Lossを計算して履歴に追加
        current_loss = self._calc_loss(model)
        self.loss_history.append(current_loss)
        
        # データの描画
        if self.ax_data is not None:
            self.ax_data.cla()
            if self.dim == 1:
                self._plot_1d(model)
            elif self.dim == 2:
                self._plot_2d(model)
            elif self.dim == 3:
                self._plot_3d(model, step)
                
        # 学習曲線の描画
        self.ax_loss.cla()
        self.ax_loss.plot(self.loss_history, color='purple', linewidth=2)
        self.ax_loss.set_title(f"Learning Curve (Loss = {current_loss:.4f})")
        self.ax_loss.set_xlabel("Iteration (Step)")
        self.ax_loss.set_ylabel("Cross Entropy Loss")
        self.ax_loss.grid(True)
        
        # 画面更新
        plt.pause(self.interval)


    # 各次元の描画ロジック 

    def _plot_1d(self, model):
        self.ax_data.set_title("1D: Sigmoid Curve")
        self.ax_data.scatter(self.explain[0], self.depend, cmap='bwr', alpha=0.6, edgecolors='k')
        
        x_min, x_max = np.min(self.explain), np.max(self.explain)
        x_line = np.linspace(x_min - 1, x_max + 1, 100)
        y_line = 1 / (1 + np.exp(-(model.w[0] * x_line + model.b)))
        
        self.ax_data.plot(x_line, y_line, color='green', linewidth=2)
        self.ax_data.grid(True)

    def _plot_2d(self, model):
        self.ax_data.set_title("2D: Decision Boundary")
        # 背景の等高線を描画
        x_min, x_max = np.min(self.explain[0])-1, np.max(self.explain[0])+1
        y_min, y_max = np.min(self.explain[1])-1, np.max(self.explain[1])+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        
        z = model.w[0]*xx + model.w[1]*yy + model.b
        p = 1 / (1 + np.exp(-z))
        self.ax_data.contourf(xx, yy, p, alpha=0.3, cmap='bwr') # 背景色
        self.ax_data.contour(xx, yy, p, levels=[0.5], colors='green', linewidths=2) # 境界線(50%)

        # 散布図
        print(self.explain)
        self.ax_data.scatter(self.explain[:, 0], self.explain[:, 1], cmap='bwr', edgecolors='k')

    def _plot_3d(self, model, step):
        self.ax_data.set_title("3D: Separating Hyperplane")
        # 3D散布図
        self.ax_data.scatter(self.explain[0], self.explain[1], self.explain[0], cmap='bwr', alpha=0.8)
        
        # 分離平面の描画
        if abs(model.w[2]) > 1e-5: # 0除算防止
            x_min, x_max = np.min(self.explain[0]), np.max(self.explain[0])
            y_min, y_max = np.min(self.explain[1]), np.max(self.explain[1])
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
            zz = -(model.w[0]*xx + model.w[1]*yy + model.b) / model.w[2]
            
            # 平面を描画
            self.ax_data.plot_surface(xx, yy, zz, color='green', alpha=0.3)
            
        # アニメーションで少しずつ回転させる
        self.ax_data.view_init(elev=20, azim=30 + step * 2)
