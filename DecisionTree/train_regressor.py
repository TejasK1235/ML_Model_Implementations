from sklearn import datasets
from sklearn.model_selection import train_test_split
from DecisionTreeRegressor import DecisionTreeRegressor
import numpy as np

def r2_score(y_pred,y_test):
    y_mean = np.mean(y_test)
    rss = np.sum((y_test-y_pred)**2)
    tss = np.sum((y_test-y_mean)**2)
    return 1 - (rss/tss)

data = datasets.load_diabetes()

X,y = data.data,data.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

model = DecisionTreeRegressor(max_depth=3,min_samples_split=100)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(r2_score(y_pred,y_test))