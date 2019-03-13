#CLUSTERING
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 1], X[y_kmeans == 0, 2], s = 30, c = 'red', label = 'iris-versicolor')
plt.scatter(X[y_kmeans == 1, 1], X[y_kmeans == 1, 2], s = 30, c = 'blue', label = 'iris-setosa')
plt.scatter(X[y_kmeans == 2, 1], X[y_kmeans == 2, 2], s = 30, c = 'green', label = 'iris-verginica')

plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s = 50, c = 'yellow', label = 'Centroids')
plt.title('Clusters of flowers')
plt.xlabel('sepel length')
plt.ylabel('sepel width')
plt.legend()
plt.show()
