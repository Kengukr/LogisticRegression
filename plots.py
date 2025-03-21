import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)

def plot_decision_boundary(model, X, y):
    x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25, 50)
    y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25, 50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_flat, yy_flat = xx.ravel(), yy.ravel()
    grid = np.c_[xx_flat, yy_flat]
    if hasattr(model, "predict_proba"):
        pred_func = model.predict_proba(grid)[:,1]
    else:
        if hasattr(model, "forward"):
            pred_func = model(torch.tensor(grid, dtype=torch.double)).detach().numpy()
        else:
            pred_func = model(grid).numpy()
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap="RdYlGn")
    plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
