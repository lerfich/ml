array = [(0, 0), (2, 4), (3, 3), (1, 2), (3, 0), (3, 1), (1, 1), (12, 18), (13, 17), (11, 15), (13, 14), (14, 16), (11, 16), (12, 15), (13, 18), (12, 5), (13, 2), (14, 4), (12, 3), (13, 1), (14, 2), (24, 19), (22, 22), (21, 24), (23, 21), (24, 20), (22, 39), (23, 38), (24, 39), (21, 37), (2, 26), (24, 6), (10, 36)]
import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering

X = np.array(array)
clusteringDef = AgglomerativeClustering(n_clusters=None, distance_threshold=4, compute_full_tree=True).fit(X)
clusteringSin = AgglomerativeClustering(linkage="single", n_clusters=None, distance_threshold=4, compute_full_tree=True).fit(X)
clusteringAve = AgglomerativeClustering(affinity="manhattan", linkage="average",n_clusters=None, distance_threshold=4, compute_full_tree=True).fit(X)
clusteringCom = AgglomerativeClustering(linkage="complete", n_clusters=None,distance_threshold=4,compute_full_tree=True).fit(X)
print(clusteringDef.labels_, '\n with Ward', clusteringDef.n_clusters_)
print(clusteringSin.labels_, '\n with Single linkage', clusteringSin.n_clusters_)
print(clusteringAve.labels_, '\n with Average linkage', clusteringAve.n_clusters_)
print(clusteringCom.labels_, '\n with Complete linkage', clusteringCom.n_clusters_)

import numpy as np



def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


# iris = load_iris()
# print(iris.data, 'dataaa')
# X = iris.data

# setting distance_threshold=0 ensures we compute the full tree.
# model = AgglomerativeClustering(distance_threshold=4, n_clusters=None, linkage="single")
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()
