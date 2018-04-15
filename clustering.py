from sys import *
import pandas as pd
from numpy import *
from copy import deepcopy
from scipy.spatial import distance_matrix

# parse parameters
file_path = argv[1]
K = int(argv[2])
model = argv[3]

# import file
data = pd.read_csv(file_path, sep=',', quotechar='"', header=0)
data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
X = data.as_matrix()

# K-means clustering algorithm
if model == "k-means":
    # Euclidean Distance Function
    def distance(x, y, ax):
        return linalg.norm(x - y, axis=ax)


    # Generate K random points as initial centroids
    n = shape(X)[1]
    centroids = mat(zeros((K, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(X[:, j])
        rangeJ = float(max(X[:, j]) - minJ)
        centroids[:, j] = mat(minJ + rangeJ * random.rand(K, 1))

    print(centroids)

    # Store old random point
    centroidsOld = zeros((K, 4))

    # cluster label for each point
    clusters = zeros(len(X))

    # Distance between new centroids and old centroids
    centroidDist = distance(centroids, centroidsOld, None)
    # print(centroidDist)

    # within-cluster sum of squared errors
    SSE = 0

    # Begin clustering
    while centroidDist != 0:
        # Assign each point to its closest cluster
        for i in range(len(X)):
            dist = distance(X[i], centroids, 1)
            # print(dist)
            clusters[i] = argmin(dist)

        # Store old centroids
        centroidsOld = deepcopy(centroids)
        # print(centroidsOld)

        # For each cluster, calculate its mean as new centroids
        for i in range(K):
            # points in the cluster
            points = []
            for j in range(len(X)):
                if clusters[j] == i:
                    points.append(X[j])
            centroids[i] = mean(points, axis=0)

            # Calculate SSE by adding up distance between points to its centroid in each cluster
            for j in range(len(points)):
                SSE += distance(points[j], centroids[i], None)

        centroidDist = distance(centroids, centroidsOld, None)

    # print(centroids)
    print("WC-SSE=%f" % SSE)
    for i in range(K):
        print("Centroid%d %s" % (i + 1, centroids[i]))

# Agglomerative Clustering
elif model == "ac":

    # calculate cluster distance
    def cluster_distance(c1, c2):
        dist_sum = 0
        for i in range(len(c1)):
            for j in range(len(c2)):
                dist_sum += linalg.norm(c1[i] - c2[j], None)
        return dist_sum / (len(c1) * len(c2))

    # initial clusters
    clusters = zeros((len(X), 4))
    for i in range(len(X)):
        clusters[i] = X[i]

    print(clusters)

    while len(clusters) > K:
        # distance matrix
        d_matrix = distance_matrix(X, X)
        print(d_matrix)

        # pairwise distance
        pair_dist = dict()
        for i in range(len(X)):
            for j in range(i):
                pair_dist[i, j] = d_matrix[j][i]
        print(pair_dist)

        # minimum distance
        mini = min(pair_dist, key=pair_dist.get)
        c1 = mini[0]
        c2 = mini[1]

        print(X[c1])
        print(X[c2])

        clusters[c1] = append([clusters[c1]], [clusters[c2]], axis=0)

        del clusters[c2]

        del d_matrix[c1]
        del d_matrix[c2]

        
else:
    print("Usage: $ python clustering.py ./some/path/file_name.csv K ac")
