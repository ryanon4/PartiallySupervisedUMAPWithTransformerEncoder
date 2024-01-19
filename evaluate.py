import sklearn
import numpy as np
from sklearn.cluster import KMeans
from functions import calculate_accuracy

def evaluate_vectors(reduced_vectors, target):
    accuracy = []
    n_clusters = len(np.unique(np.array(target)))

    for vectors in reduced_vectors:
        try:
            dimensionality = len(vectors[0])
            print("Dimensionality: " + str(dimensionality))

            cluster_model = KMeans(n_clusters=n_clusters, n_init=200, init="random", max_iter=1000).fit(vectors)

            clusters = cluster_model.labels_

            accuracy.append(calculate_accuracy(target, clusters))
        except Exception as e:
            continue
    return accuracy

def evaluate_vectors_baseline(vectors, target):
    n_clusters = len(np.unique(np.array(target)))


    dimensionality = len(vectors[0])
    print("Dimensionality: " + str(dimensionality))
    cluster_model = KMeans(n_clusters=n_clusters, n_init=200, init="random", max_iter=1000).fit(vectors)
    clusters = cluster_model.labels_
    accuracy = calculate_accuracy(target, clusters)

    return accuracy