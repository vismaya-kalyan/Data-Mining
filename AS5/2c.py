# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 19:02:16 2018

@author: visma
"""

"""
====================================
Demonstration of k-means assumptions
====================================

This example is meant to illustrate situations where k-means will produce
unintuitive and possibly unexpected clusters. In the first three plots, the
input data does not conform to some implicit assumption that k-means makes and
undesirable clusters are produced as a result. In the last plot, k-means
returns intuitive clusters despite unevenly sized blobs.
"""
print(__doc__)

# Author: Phil Roth <mr.phil.roth@gmail.com>
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

#plt.figure(figsize=(12, 12))

n_samples = 1500
random_state = 170
X, labels_true = make_blobs(n_samples=n_samples, random_state=random_state)


# Anisotropicly distributed data
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
#X_aniso = np.dot(X, transformation)
#X = np.dot(X, transformation) #change here

# Different variance
#X_varied, y_varied = make_blobs(n_samples=n_samples,
#                                cluster_std=[1.0, 2.5, 0.5],
#                                random_state=random_state)

#X, labels_true = make_blobs(n_samples=n_samples,
#                                cluster_std=[1.0, 2.5, 0.5],
#                                random_state=random_state)
#change above

# Unevenly sized blobs
#X_filtered = np.vstack((X[y == 0][:500], X[labels_true == 1][:100], X[labels_true == 2][:10]))
X = np.vstack((X[labels_true == 0][:500], X[labels_true == 1][:100], X[labels_true == 2][:10])) #change here
l5 = labels_true[labels_true == 0]
l5 = l5[:500]

l100 = labels_true[labels_true == 1]
l100 = l100[:100]

l10 = labels_true[labels_true == 2]
l10 = l10[:10]

labels_true = np.vstack((l5.reshape(500,1), l100.reshape(100,1), l10.reshape(10,1))) #change here
labels_true = labels_true[:610,0]

#X = StandardScaler().fit_transform(X)

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.4, min_samples=10).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result
#import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
