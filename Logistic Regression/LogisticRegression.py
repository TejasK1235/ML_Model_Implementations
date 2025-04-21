import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegression:
    def __init__(self,lr=0.0001,epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(Self,X,y):
        m, n = X.shape
        Self.b = 0
        Self.w = np.zeros(n)

        for _ in range(Self.epochs):
            z_wb = np.dot(X,Self.w) + Self.b
            f_wb = sigmoid(z_wb)

            dj_dw = (1/m) * np.dot(X.T, (f_wb - y))
            dj_db = (1/m) * np.sum(f_wb - y)

            #Gradient descent
            Self.w -= Self.lr * dj_dw
            Self.b -= Self.lr * dj_db

    def predict(Self,X):
        z_wb = np.dot(X,Self.w) + Self.b
        f_wb = sigmoid(z_wb)

        y_pred = [0 if y <= 0.5 else 1 for y in f_wb]
        return y_pred
