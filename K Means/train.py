from sklearn.datasets import make_blobs
import numpy as np
from KMeans import KMeans

X,y = make_blobs(centers=3,n_samples=500,n_features=2,shuffle=True,random_state=42)

print(X.shape)

clusters = len(np.unique(y))
print(clusters)

model = KMeans(K=clusters,plot_steps=True)
y_pred = model.predict(X)

model.plot()