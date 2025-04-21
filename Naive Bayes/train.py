from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from NaiveBayes import NaiveBayes

def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test) / len(y_test)

X,y = datasets.make_classification(n_samples=100,n_features=10,n_classes=2,random_state=123)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

model = NaiveBayes()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(accuracy(y_pred,y_test))
