import numpy as np

class LinearRegression:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(Self,X,y):
        m, n = X.shape
        Self.b = 0
        Self.w = np.zeros(n)

        for _ in range(Self.epochs):
            f_wb = np.dot(X,Self.w) + Self.b

            dj_dw = (1/m) * np.dot(X.T, (f_wb - y))
            dj_db = (1/m) * np.sum(f_wb - y)

            #Gradient descent
            Self.w -= Self.lr * dj_dw
            Self.b -= Self.lr * dj_db

    def predict(Self,X):
        f_wb = np.dot(X,Self.w) + Self.b
        return f_wb
