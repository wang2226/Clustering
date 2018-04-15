from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np


file_path = "./yelp.csv"
#file_path = "./dummy.csv"

data = pd.read_csv(file_path, sep=',', quotechar='"', header=0)
data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]

latitude = data['latitude'].values
longitude = data['longitude'].values
rc = data['reviewCount'].values
#rc = np.log(rc)
checkins = data['checkins'].values
#checkins = np.log(checkins)

X = np.array(list(zip(latitude, longitude)))
#X_scaled = scale(X)

K = 4

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

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='*', s=200, alpha=0.5)
plt.title("Latitude vs. Longitude")
plt.savefig('Latitude-Longitude.jpg')
#plt.title("ReviewCount vs. Check-ins (scaled)")
#plt.savefig('rc vs. checkins (scaled).jpg')
