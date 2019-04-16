# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:05:01 2018

@author: visma
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

np.random.seed(0)


n_samples = 1500


blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
#no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)



X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
filtered = (X_filtered, y)

plt.figure(figsize=(5, 20))
plt.gca().set_aspect('equal', adjustable='box')

plot_num = 1

default_base = {
#                'quantile': .3,
                'eps': .1,
#                'damping': .9,
#                'preference': -200,
                'n_neighbors': 10
#                'n_clusters': 3
                }



datasets = [
    (varied, {'eps': .18, 'n_neighbors': 2}),
    (aniso, {'eps': .15, 'n_neighbors': 2}),
    (blobs, {}),
    (filtered, {})
    ]


for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)


    dbscan = cluster.DBSCAN(eps=params['eps'])

    dbscan.fit(X)

     
    if hasattr(dbscan, 'labels_'):
        y_pred = dbscan.labels_.astype(np.int)
    else:
        y_pred = dbscan.predict(X)

    plt.subplot(len(datasets), 1, plot_num)
    if i_dataset == 0:
        plt.title('DBSCAN', size=18)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())

    plot_num += 1

plt.show()



