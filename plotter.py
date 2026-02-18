import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, interval, explain, depend):
        self.interval = interval
        self.explain = explain
        self.depend = depend
        self.dim = explain.shape[1]
        self.loss_history = []
        
        self.fig = plt.figure(figsize=(12, 6)) 
        
        if self.dim == 1:
            self.ax_data = self.fig.add_subplot(1, 2, 1)
        elif self.dim == 2:
            self.ax_data = self.fig.add_subplot(1, 2, 1)
        elif self.dim == 3:
            self.ax_data = self.fig.add_subplot(1, 2, 1, projection='3d')
        else:
            self.ax_data = None 
            
        if self.dim <= 3:
            self.ax_loss = self.fig.add_subplot(1, 2, 2)
        else:
            self.ax_loss = self.fig.add_subplot(1, 1, 1)

    def _calc_loss(self, models):
        total_loss = 0
        for i, model in enumerate(models):
            # 正解ラベル
            y_ovr = np.where(self.depend == i, 1, 0)
            
            z = model.w @ self.explain.T + model.b
            p = 1 / (1 + np.exp(-z))
            epsilon = 1e-15
            loss = -np.mean(y_ovr * np.log(p + epsilon) + (1 - y_ovr) * np.log(1 - p + epsilon))
            total_loss += loss
        return total_loss / len(models) 

    # 【修正】引数を models (リスト) に変更
    def show(self, models, step):
        current_loss = self._calc_loss(models)
        self.loss_history.append(current_loss)
        
        if self.ax_data is not None:
            self.ax_data.cla()
            if self.dim == 1:
                self._plot_1d(models)
            elif self.dim == 2:
                self._plot_2d(models)
            elif self.dim == 3:
                self._plot_3d(models, step)
                
        self.ax_loss.cla()
        self.ax_loss.plot(self.loss_history, color='purple', linewidth=2)
        self.ax_loss.set_title(f"OvR Learning Curve (Average Loss = {current_loss:.4f})")
        self.ax_loss.set_xlabel("Iteration (Step)")
        self.ax_loss.set_ylabel("Cross Entropy Loss")
        self.ax_loss.grid(True)
        
        plt.pause(self.interval)


    def _plot_1d(self, models):
        self.ax_data.set_title("1D: OvR Sigmoid Curves")
        cmap = plt.get_cmap('tab10')
        
        self.ax_data.scatter(self.explain[:, 0], self.depend, c=self.depend, cmap=cmap, alpha=0.6, edgecolors='k')
        
        x_min, x_max = np.min(self.explain[:, 0]), np.max(self.explain[:, 0])
        x_line = np.linspace(x_min - 1, x_max + 1, 100)
        
        # モデルごとのシグモイド曲線を重ねて描画
        for i, model in enumerate(models):
            y_line = 1 / (1 + np.exp(-(model.w[0] * x_line + model.b)))
            self.ax_data.plot(x_line, y_line, color=cmap(i), linewidth=2, label=f'Class {i} Prob')
        
        self.ax_data.grid(True)
        self.ax_data.legend()

    def _plot_2d(self, models):
        self.ax_data.set_title("2D: OvR Decision Regions")
        cmap = plt.get_cmap('tab10', len(models))
        
        x_min, x_max = np.min(self.explain[:, 0])-1, np.max(self.explain[:, 0])+1
        y_min, y_max = np.min(self.explain[:, 1])-1, np.max(self.explain[:, 1])+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        # 座標リストを作成
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # すべてのモデルに確率を計算させる
        probs = []
        for model in models:
            z = grid_points @ model.w + model.b
            p = 1 / (1 + np.exp(-z))
            probs.append(p)
        
        # 一番確率が高かったモデルのクラスを取得 
        probs = np.array(probs)
        predictions = np.argmax(probs, axis=0)
        predictions = predictions.reshape(xx.shape)
        
        self.ax_data.contourf(xx, yy, predictions, alpha=0.3, cmap=cmap)
        
        # 実際のデータを描画
        self.ax_data.scatter(self.explain[:, 0], self.explain[:, 1], c=self.depend, cmap=cmap, edgecolors='k')

    def _plot_3d(self, models, step):
        self.ax_data.set_title("3D: OvR Hyperplanes")
        cmap = plt.get_cmap('tab10', len(models))
        
        self.ax_data.scatter(self.explain[:, 0], self.explain[:, 1], self.explain[:, 2], c=self.depend, cmap=cmap, alpha=0.8, edgecolors='k')
        
        for i, model in enumerate(models):
            if abs(model.w[2]) > 1e-5:
                x_min, x_max = np.min(self.explain[:, 0]), np.max(self.explain[:, 0])
                y_min, y_max = np.min(self.explain[:, 1]), np.max(self.explain[:, 1])
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
                zz = -(model.w[0]*xx + model.w[1]*yy + model.b) / model.w[2]
                
                z_min, z_max = np.min(self.explain[:, 2])-2, np.max(self.explain[:, 2])+2
                zz = np.clip(zz, z_min, z_max)
                
                self.ax_data.plot_surface(xx, yy, zz, color=cmap(i), alpha=0.2) # 各クラスの色で平面を描く
                self.ax_data.set_zlim(z_min, z_max)
                
        self.ax_data.view_init(elev=20, azim=30 + step * 2)