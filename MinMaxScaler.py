import numpy as np


class MinMaxScaler:
    def __init__(self):
        self.min = 0
        self.max = 0
        self.scale = 0

    def fit_transform(self, X):
        self.min = np.min(X)
        self.max = np.max(X)
        self.scale = 1 / (self.max - self.min)
        return self.transform(X)

    def transform(self, X):
        return (X - self.min) * self.scale

    def inverse_transform(self, X):
        return (X / self.scale) + self.min

    def __repr__(self) -> str:
        return f"MinMaxScaler(min={self.min}, max={self.max}, scale={self.scale})"

    def __str__(self) -> str:
        return self.__repr__()
