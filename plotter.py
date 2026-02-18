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

    def show(self, model, step):
        
        if self.ax_data is not None:
            self.ax_data.cla()
            if self.dim == 1:
                self._plot_1d(model)
            elif self.dim == 2:
                self._plot_2d(model)
            elif self.dim == 3:
                self._plot_3d(model, step)
                
        self.ax_loss.cla()
        self.ax_loss.plot(model.loss_history, color='purple', linewidth=2)
        self.ax_loss.set_title(f"Learning Curve")
        self.ax_loss.set_xlabel("Iteration (Step)")
        self.ax_loss.set_ylabel("Cross Entropy Loss")
        self.ax_loss.grid(True)
        
        plt.pause(self.interval)

    def _plot_2d(self, model):
        self.ax_data.set_title("2D: Decision Boundary")
        x_min, x_max = np.min(self.explain[:, 0])-1, np.max(self.explain[:, 0])+1
        y_min, y_max = np.min(self.explain[:, 1])-1, np.max(self.explain[:, 1])+1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        Z_grid = grid_points @ model.W + model.b
        Z_max_grid = np.max(Z_grid, axis=1, keepdims=True)
        exp_Z_grid = np.exp(Z_grid - Z_max_grid)
        P_grid = exp_Z_grid / np.sum(exp_Z_grid, axis=1, keepdims=True)
        
        predicted_grid = np.argmax(P_grid, axis=1)
        predicted_grid = predicted_grid.reshape(xx.shape)

        self.ax_data.contourf(xx, yy, predicted_grid, alpha=0.3, cmap='bwr')
        
        self.ax_data.scatter(self.explain[:, 0], self.explain[:, 1], c=self.depend, cmap='bwr', edgecolors='k')

    def _plot_1d(self, model):
        self.ax_data.set_title("1D: Class Probabilities")
        # 実際のデータをプロット
        self.ax_data.scatter(self.explain[:, 0], self.depend, c=self.depend, cmap='brg', alpha=0.8, edgecolors='k')
        
        x_min, x_max = np.min(self.explain[:, 0]), np.max(self.explain[:, 0])
        
        x_line = np.linspace(x_min - 1, x_max + 1, 100).reshape(-1, 1)
        
        # 予測確率
        Z_line = x_line @ model.W + model.b
        Z_max_line = np.max(Z_line, axis=1, keepdims=True)
        exp_Z_line = np.exp(Z_line - Z_max_line)
        P_line = exp_Z_line / np.sum(exp_Z_line, axis=1, keepdims=True)
        
        colors = ['blue', 'red', 'green']
        for k in range(model.W.shape[1]):
            self.ax_data.plot(x_line, P_line[:, k], color=colors[k % len(colors)], linewidth=2, label=f'Class {k}')
        
        self.ax_data.grid(True)
        self.ax_data.legend(loc='upper right')

    def _plot_3d(self, model, step):
        self.ax_data.set_title("3D: Decision Regions")
        # 実際のデータをプロット
        self.ax_data.scatter(self.explain[:, 0], self.explain[:, 1], self.explain[:, 2], 
                             c=self.depend, cmap='brg', alpha=1.0, edgecolors='k', s=40)
        
        x_min, x_max = np.min(self.explain[:, 0])-1, np.max(self.explain[:, 0])+1
        y_min, y_max = np.min(self.explain[:, 1])-1, np.max(self.explain[:, 1])+1
        z_min, z_max = np.min(self.explain[:, 2])-1, np.max(self.explain[:, 2])+1

        grid_res = 15
        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                                 np.linspace(y_min, y_max, grid_res),
                                 np.linspace(z_min, z_max, grid_res))
        
        grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        
        # 空間上の点に対する予測確率
        Z_grid = grid_points @ model.W + model.b
        Z_max_grid = np.max(Z_grid, axis=1, keepdims=True)
        exp_Z_grid = np.exp(Z_grid - Z_max_grid)
        P_grid = exp_Z_grid / np.sum(exp_Z_grid, axis=1, keepdims=True)
        
        predicted_grid = np.argmax(P_grid, axis=1)
        
        self.ax_data.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2],
                             c=predicted_grid, cmap='brg', alpha=0.1, marker='s', s=10)
            
        self.ax_data.view_init(elev=20, azim=30 + step * 2)