import matplotlib.pyplot as plt
import numpy as np

class Plotter:
    def __init__(self, interval, explain, depend):
        self.interval = interval
        self.explain = explain
        self.depend = depend
        
        # データの次元数 を取得 (100, 2) なら 2 になる
        self.dim = explain.shape[1]

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
            self.ax_data = None 
            
        if self.dim <= 3:
            self.ax_loss = self.fig.add_subplot(1, 2, 2)
        else:
            self.ax_loss = self.fig.add_subplot(1, 1, 1)

    def _calc_loss(self, model):
        # explain.T で (次元数, データ数) に変換してから内積をとる（完璧です！）
        z = model.w @ self.explain.T + model.b
        p = 1 / (1 + np.exp(-z))
        epsilon = 1e-15
        loss = -np.mean(self.depend * np.log(p + epsilon) + (1 - self.depend) * np.log(1 - p + epsilon))
        return loss

    def show(self, model, step):
        current_loss = self._calc_loss(model)
        self.loss_history.append(current_loss)
        
        if self.ax_data is not None:
            self.ax_data.cla()
            if self.dim == 1:
                self._plot_1d(model)
            elif self.dim == 2:
                self._plot_2d(model)
            elif self.dim == 3:
                self._plot_3d(model, step)
                
        self.ax_loss.cla()
        self.ax_loss.plot(self.loss_history, color='purple', linewidth=2)
        self.ax_loss.set_title(f"Learning Curve (Loss = {current_loss:.4f})")
        self.ax_loss.set_xlabel("Iteration (Step)")
        self.ax_loss.set_ylabel("Cross Entropy Loss")
        self.ax_loss.grid(True)
        
        plt.pause(self.interval)


    def _plot_1d(self, model):
        self.ax_data.set_title("1D: Sigmoid Curve")
        self.ax_data.scatter(self.explain[:, 0], self.depend, c=self.depend, cmap='bwr', alpha=0.6, edgecolors='k')
        
        x_min, x_max = np.min(self.explain[:, 0]), np.max(self.explain[:, 0])
        x_line = np.linspace(x_min - 1, x_max + 1, 100)
        y_line = 1 / (1 + np.exp(-(model.w[0] * x_line + model.b)))
        
        self.ax_data.plot(x_line, y_line, color='green', linewidth=2)
        self.ax_data.grid(True)

    def _plot_2d(self, model):
        self.ax_data.set_title("2D: Decision Boundary")
        x_min, x_max = np.min(self.explain[:, 0])-1, np.max(self.explain[:, 0])+1
        y_min, y_max = np.min(self.explain[:, 1])-1, np.max(self.explain[:, 1])+1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        
        z = model.w[0]*xx + model.w[1]*yy + model.b
        p = 1 / (1 + np.exp(-z))
        self.ax_data.contourf(xx, yy, p, alpha=0.3, cmap='bwr')
        self.ax_data.contour(xx, yy, p, levels=[0.5], colors='green', linewidths=2)

        self.ax_data.scatter(self.explain[:, 0], self.explain[:, 1], c=self.depend, cmap='bwr', edgecolors='k')

    def _plot_3d(self, model, step):
        self.ax_data.set_title("3D: Separating Hyperplane")
        self.ax_data.scatter(self.explain[:, 0], self.explain[:, 1], self.explain[:, 2], c=self.depend, cmap='bwr', alpha=0.8, edgecolors='k')
        
        if abs(model.w[2]) > 1e-5:
            x_min, x_max = np.min(self.explain[:, 0]), np.max(self.explain[:, 0])
            y_min, y_max = np.min(self.explain[:, 1]), np.max(self.explain[:, 1])
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
            zz = -(model.w[0]*xx + model.w[1]*yy + model.b) / model.w[2]
            
            z_min, z_max = np.min(self.explain[:, 2])-2, np.max(self.explain[:, 2])+2
            zz = np.clip(zz, z_min, z_max)
            
            self.ax_data.plot_surface(xx, yy, zz, color='green', alpha=0.3)
            self.ax_data.set_zlim(z_min, z_max)
            
        self.ax_data.view_init(elev=20, azim=30 + step * 2)