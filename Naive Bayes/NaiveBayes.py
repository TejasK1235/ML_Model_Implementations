import numpy as np

class NaiveBayes:
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self._labels = np.unique(y)

        n_labels = len(self._labels)

        # calcualte mean,var and prior for each label
        self._mean = np.zeros((n_labels,n_features))
        self._var = np.zeros((n_labels,n_features))
        self._prior = np.zeros(n_labels)

        for idx,label in enumerate(self._labels):
            X_label = X[y==label]
            self._mean[idx,:] = X_label.mean(axis=0)
            self._var[idx,:] = X_label.var(axis=0)
            self._prior[idx] = X_label.shape[0] / n_samples

    def predict(self,X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self,x):
        posteriors = []

        for idx,label in enumerate(self._labels):
            prior = np.log(self._prior[idx])
            posterior = np.sum(np.log(self._guassian(idx,x)))
            posterior += prior
            posteriors.append(posterior)

        return self._labels[np.argmax(posteriors)]
    
    def _guassian(self,label_idx,x):
        mean = self._mean[label_idx]
        var = self._var[label_idx]
        numerator = np.exp(-((x - mean)**2) / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator