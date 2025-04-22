import numpy as np
import matplotlib.pyplot as plt

def euclidean(x,y):
    return np.sqrt(np.sum((x-y)**2))

class KMeans:
    def __init__(self,K=5,max_iters=100,plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        self.clusters = [[] for _ in range(self.K)]

        self.centroids = []

    def predict(self,X):
        self.X = X
        self.n_samples,self.n_features = X.shape
        
        # create initial centroids
        random_sample_idxs = np.random.choice(self.n_samples,self.K,replace=False)
        self.centroids = [self.X[i] for i in random_sample_idxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # find new centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old,self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # assign samples to new centroids
        return self._get_cluster_labels(self.clusters)
    
    def _get_cluster_labels(self, clusters):
        # assign each sample to a cluster
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _create_clusters(self, centroids):
        # assign the samples to nearest centroid
        clusters = [[] for _ in range(self.K)]
        for idx,sample in enumerate(self.X):
            centroid_idx = self._nearest_centroid(sample,centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _nearest_centroid(self,sample,centroids):
        # find minimum distance between sample and all centroids
        distances = [euclidean(sample,centroid) for centroid in centroids]
        nearest_idx = np.argmin(distances)
        return nearest_idx

    def _get_centroids(self, clusters):
        # assign mean of clusters to centroids
        centroids = np.zeros((self.K,self.n_features))
        for cluster_idx,cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster],axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self,centroids_old,centroids):
        # check is old and new centroids are the same
        distances = [euclidean(centroids_old[i],centroids[i]) for i in range(self.K)]
        return sum(distances) == 0


    def plot(self):
        fig,ax = plt.subplots()

        for cluster in self.clusters:
            points = self.X[cluster].T
            ax.scatter(points[0],points[1])

        for point in self.centroids:
            ax.scatter(point[0],point[1],marker="X",color="black",linewidth=2)
        
        plt.show()


