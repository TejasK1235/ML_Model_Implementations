import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from LogisticRegression import LogisticRegression

def accuracy(y_pred,y_test):
    return np.sum(y_pred == y_test)/len(y_test)


X,y = make_blobs(n_features=10,n_samples=100,centers=2,cluster_std=3)

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1234,test_size=0.2)


model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy(y_pred,y_test))


