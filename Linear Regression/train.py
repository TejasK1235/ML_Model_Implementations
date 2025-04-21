import numpy as np
from sklearn.datasets import make_regression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

def mse(y_pred,y_test):
    return np.mean((y_pred-y_test)**2)

def r2_score(y_pred,y_test):
    y_mean = np.mean(y_test)
    rss = np.sum((y_test - y_pred)**2)
    tss = np.sum((y_test - y_mean)**2)
    return 1 - (rss/tss)

X, y = make_regression(n_features=2,n_samples=100,noise=50)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1234,test_size=0.2)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(mse(y_pred,y_test))
print(r2_score(y_pred,y_test))