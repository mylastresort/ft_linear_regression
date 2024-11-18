import numpy as np


class LinearRegression:
    w = 0
    b = 0

    def __init__(self, lr=0.001, n_iters=1000) -> None:
        self.lr = lr
        self.n = n_iters

    def set_parameters(self, w: float, b: float) -> None:
        self.w = w
        self.b = b

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x * self.w + self.b

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        n_samples = len(x)
        for _ in range(self.n):
            y_pred = self.predict(x)
            _weight = (1 / n_samples) * np.dot(x, y_pred - y)
            _bias = (1 / n_samples) * np.sum(y_pred - y)
            self.w -= self.lr * _weight
            self.b -= self.lr * _bias

    def cost(self, x: np.ndarray, y: np.ndarray):
        y_pred = self.predict(x)
        n_samples = len(x)
        return np.sum((y - y_pred) ** 2) / (n_samples * 2)
