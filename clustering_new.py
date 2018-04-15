import sys
import pandas as pd
import random
from numpy import *
from copy import deepcopy
import heapq
import itertools


# import file
def load_data(file_path):
    data = pd.read_csv(file_path, sep=',', quotechar='"', header=0)
    data = data[['latitude', 'longitude', 'reviewCount', 'checkins']]
    return data.as_matrix()


# Euclidean Distance Function
def eclud_distance(x, y, ax):
    return linalg.norm(x - y, axis=ax)


# Generate K random points as initial centroids
def rand_centroids(data_set, k):
    n = shape(data_set)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        min_j = min(data_set[:, j])
        range_j = float(max(data_set[:, j]) - min_j)
        centroids[:, j] = mat(min_j + range_j * random.rand(k, 1))
    return centroids
    # print(centroids)


# K-means clustering algorithm
def k_means(data_set, k):
    # within-cluster sum of squared errors
    SSE = 0
    # Begin clustering
    m = shape(data_set)[0]
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = rand_centroids(data_set, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = Inf
            minIndex = -1
            for j in range(k):
                distJI = eclud_distance(centroids[j, :], data_set[i, :], ax=1)
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2

        for cent in range(k):  # recalculate centroids
            ptsInClust = data_set[nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean

    # print(centroids)
    print("WC-SSE=%f" % SSE)
    for i in range(K):
        print("Centroid%d %s" % (i + 1, centroids[i]))


# calculate cluster distance
def cluster_distance(c1, c2):
    return sum([pairwise_matrix[x1][x2]
                for x1 in c1
                for x2 in c2]) \
           / (len(c1) * len(c2))


# calculate pairwise distance
def pairwise_distance(data_set):
    result = []
    dataset_size = len(data_set)

    for i in range(dataset_size - 1):  # ignore last i
        for j in range(i + 1, dataset_size):  # ignore duplication
            distance = eclud_distance(data_set[i]["data"], data_set[j]["data"], None)
            pairwise_matrix[i][j] = distance
            pairwise_matrix[j][i] = distance

            # duplicate dist, need to be remove, and there is no difference to use tuple only
            # leave second dist here is to take up a position for tie selection
            result.append((distance, [distance, [[i], [j]]]))

    return result


def compute_centroid(dataset, data_points_index, dimension):
    size = len(data_points_index)
    dim = dimension
    centroid = [0.0] * dim
    for idx in data_points_index:
        dim_data = dataset[idx]["data"]
        for i in range(dim):
            centroid[i] += float(dim_data[i])
    for i in range(dim):
        centroid[i] /= size
    return centroid


def data_dictionary(data_matrix):
    data_set = []
    clusters = {}
    id = 0

    for row in data_matrix:
        data = {}
        data.setdefault("id", id)  # duplicate
        data.setdefault("data", row)
        data_set.append(data)

        clusters_key = str([id])
        clusters.setdefault(clusters_key, {})
        clusters[clusters_key].setdefault("centroid", row)
        clusters[clusters_key].setdefault("elements", [id])

        id += 1
    return data_set, clusters


def build_priority_queue(distance_list):
    heapq.heapify(distance_list)
    heap = distance_list
    return heap


def valid_heap_node(heap_node, old_clusters):
    # pair_dist = heap_node[0]
    pair_data = heap_node[1]
    for old_cluster in old_clusters:
        if old_cluster in pair_data:
            return False
    return True


def add_heap_entry(heap, new_cluster, current_clusters):
    new_heap_entry = []
    min_dist = Inf
    dist = 0.0
    for ex_cluster in current_clusters.values():
        dist = cluster_distance(ex_cluster["elements"], new_cluster["elements"])
        if dist < min_dist:
            min_dist = dist
            new_heap_entry.clear()
            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
    heapq.heappush(heap, (dist, new_heap_entry))


# Agglomerative Clustering
def agglomerative_clustering(data_set, clusters, k, dimension):
    current_clusters = clusters
    old_clusters = []
    heap = pairwise_distance(data_set)
    heap = build_priority_queue(heap)

    while len(current_clusters) > k:
        dist, min_item = heapq.heappop(heap)
        # pair_dist = min_item[0]
        pair_data = min_item[1]
        # judge if include old cluster
        if not valid_heap_node(min_item, old_clusters):
            continue

        new_cluster = {}
        # new_cluster_elements = sum(pair_data, [])
        new_cluster_elements = list(itertools.chain.from_iterable(pair_data))
        new_cluster_cendroid = compute_centroid(data_set, new_cluster_elements, dimension)
        new_cluster_elements.sort()
        new_cluster.setdefault("centroid", new_cluster_cendroid)
        new_cluster.setdefault("elements", new_cluster_elements)
        for pair_item in pair_data:
            old_clusters.append(pair_item)
            del current_clusters[str(pair_item)]
        add_heap_entry(heap, new_cluster, current_clusters)
        current_clusters[str(new_cluster_elements)] = new_cluster

    # Calculate SSE by adding up distance between points to its centroid in each cluster
    SSE = 0.0
    centroids = []

    for key in current_clusters:
        centroid = current_clusters[key]["centroid"]
        data_points_index = current_clusters[key]["elements"]
        for idx in data_points_index:
            dim_data = data_set[idx]["data"]
            SSE += eclud_distance(dim_data, centroid, None)
            centroids.append(centroid)

    # print(centroids)
    print("WC-SSE=%f" % SSE)
    i = 0
    for key in current_clusters:
        i += 1
        print("Centroid%d %s" % (i, current_clusters[key]["centroid"]))

    return current_clusters


# parse parameters
file_path = sys.argv[1]
K = int(sys.argv[2])
model = sys.argv[3]

X = load_data(file_path)

if model == "km":
    k_means(X, K)
elif model == "ac":
    data_set, clusters = data_dictionary(X)
    pairwise_matrix = zeros([len(X), len(X)])
    dimension = len(data_set[0]["data"])
    agglomerative_clustering(data_set, clusters, K, dimension)
else:
    print("Usage: $ python clustering.py ./some/path/file_name.csv K ac")
