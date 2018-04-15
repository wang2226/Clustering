from sklearn.cluster import KMeans
import pandas as pd

file_path = "./yelp.csv"

data = pd.read_csv(file_path, sep=',', quotechar='"', header=0)
data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
X = data.as_matrix()

K = int(input("Enter value for K: \n"))

# Number of clusters
kmeans = KMeans(n_clusters=K)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_

print("within-cluster sum of squares: %f" %kmeans.inertia_)

# Scikit-learn centroids
for i in range(K):
    print("Centroid%d %s" % (i + 1, centroids[i]))
