import numpy as np

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)
        self.n_labels_ = n_samples
        
        while self.n_labels_ > self.n_clusters:
            distances = self._compute_distances(X)
            min_dist = np.argmin(distances)
            i, j = np.unravel_index(min_dist, (self.n_labels_, self.n_labels_))
            
            # Merge clusters i and j
            self._merge(i, j)
        
        self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            self.cluster_centers_[k] = np.mean(X[self.labels_ == k], axis=0)
        
        return self
        
    def _compute_distances(self, X):
        distances = np.zeros((self.n_labels_, self.n_labels_))
        for i in range(self.n_labels_):
            for j in range(i+1, self.n_labels_):
                if self.linkage == 'single':
                    dist = np.min(np.sqrt(np.sum((X[self.labels_ == i, np.newaxis, :] - X[self.labels_ == j, :])**2, axis=2)))
                elif self.linkage == 'complete':
                    dist = np.max(np.sqrt(np.sum((X[self.labels_ == i, np.newaxis, :] - X[self.labels_ == j, :])**2, axis=2)))
                elif self.linkage == 'average':
                    dist = np.mean(np.sqrt(np.sum((X[self.labels_ == i, np.newaxis, :] - X[self.labels_ == j, :])**2, axis=2)))
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _merge(self, i, j):
        mask = self.labels_ == j
        self.labels_[mask] = i
        self.n_labels_ -= 1
        
        
        

def format_distance_matrix(dist_matrix, labels=None, num_decimals=3):
    n = len(dist_matrix)
    formatted_matrix = np.zeros((n, n), dtype='U50')
    
    # Add labels to rows and columns
    if labels is not None:
        if len(labels) != n:
            raise ValueError("Number of labels must match number of rows/columns in distance matrix.")
        for i in range(n):
            formatted_matrix[i, 0] = formatted_matrix[0, i] = labels[i]
    
    # Format distances with specified number of decimals
    for i in range(n):
        for j in range(i+1, n):
            formatted_matrix[i, j] = formatted_matrix[j, i] = '{:.{}f}'.format(dist_matrix[i, j], num_decimals)
    
    return formatted_matrix


if __name__ == "__main__":
    
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, y = make_blobs(n_samples=100, centers=3, random_state=42)

    model = AgglomerativeClustering(n_clusters=3, linkage='single')
    model.fit(X)

    plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker='*', s=200, c='r')
    plt.show()

    
    # from distance matrix 
    
    # Generate random distance matrix
    np.random.seed(42)
    dist_matrix = np.random.rand(5, 5)

    # Format distance matrix with 2 decimal places
    formatted_matrix = format_distance_matrix(dist_matrix, labels=['A', 'B', 'C', 'D', 'E'], num_decimals=2)

    # Print formatted matrix
    print(formatted_matrix)
