from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

#file_path = "./yelp.csv"
file_path = "./dummy.csv"

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

K = 3

# AC Clustering
ac = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="average")
ac.fit(X)
labels = ac.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

plt.title("Latitude vs. Longitude (ac)")
plt.savefig('Latitude-Longitude (ac).jpg')
#plt.title("ReviewCount vs. Check-ins (scaled)")
#plt.savefig('rc vs. checkins (scaled).jpg')
