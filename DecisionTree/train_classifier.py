from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTreeClassifier import DecisionTreeClassifier
import numpy as np

def accuracy(y_pred,y_test):
    return np.sum(y_pred==y_test)/len(y_pred)



data = datasets.load_breast_cancer()

X,y = data.data,data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(accuracy(y_pred,y_test))