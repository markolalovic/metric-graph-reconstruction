#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" baseline.py: Functions and classes for the baseline approach. """

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import MeanShift
from sklearn.neighbors import KernelDensity
from sklearn.cluster import estimate_bandwidth

def save_data(name, data):
    path = "../data/" + name + ".xy"
    np.savetxt(path, data)

def load_data(name):
    path = "../data/" + name + ".xy"
    data = np.loadtxt(path)
    return data

def simulate_data():
    n_samples = 100
    modes_sim = [(1,1), (3,1)]
    k_sim = len(modes_sim)
    data, labels_sim = make_blobs(n_samples=n_samples,
                                  centers=modes_sim,
                                  n_features=k_sim,
                                  center_box=(0, 1),
                                  cluster_std = 0.30)
    save_data("test_data", data)
    save_data("test_labels", labels_sim)

def visualize(data):
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, c="blue")
    plt.show()

def visualize_sim(data, labels):
    colors = 10*['r.','g.', 'b.', 'c.','m.','y.']
    for i in range(data.shape[0]):
        plt.plot(data[i][0], data[i][1],
                 colors[labels[i]], markersize=10, alpha=0.3)
    plt.show()

def normalize(data):
    d_min = np.min(data, axis=0)
    d_max = np.max(data, axis=0)
    data_n = (data - d_min) / (d_max - d_min)
    return data_n

def prepare_grid(a, b, c, d, n=100):
    x = np.linspace(a, b, n)
    y = np.linspace(c, d, n)
    X, Y = np.meshgrid(x, y)
    xy = np.vstack([Y.ravel(), X.ravel()]).T
    return X, Y, xy

def kernel_bandwidth(data):
    """ Normal reference rule with a slight modification of Chacon et al. (2011).
    TODO: on small data set via grid search cross-validation to balance bias-variance
    """
    d = data.shape[1]
    n = data.shape[0]
    s = 1/d * np.sum(np.std(data, axis=0))
    h = s * (4 / (d + 4))**(1/(d + 6)) * n**(-1/(d + 6))
    return h

def kernel_bandwidth_sklearn(data, quantile=0.3, n_samples=None):
    """ TODO: try elbow method by increasing quantile to determine the number of modes. """
    return estimate_bandwidth(data, quantile=quantile, n_samples=n_samples)

def evaluate_density_sklearn(xy, data, h):
    """ TODO: increase n_jobs to speed it up. """
    kde = KernelDensity(bandwidth=h, kernel='gaussian')
    kde.fit(data)
    Z = np.exp(kde.score_samples(xy))
    return Z

def kernel(x):
    return np.exp(-x**2 / 2.)

def distance(x, y):
    return np.linalg.norm(x - y)

def soft_assignment(data, modes, h):
    n = data.shape[0]
    k = modes.shape[0]

    S = np.zeros((n, k))
    T = np.zeros((n, n))
    for i in range(n):
        nrmc = np.sum(kernel(np.linalg.norm(data[i] - data, axis=1)/h))
        nrmc += np.sum(kernel(np.linalg.norm(data[i] - modes, axis=1)/h))
        S[i, :] = kernel(np.linalg.norm(data[i] - modes, axis=1)/h)/nrmc
        T[i, :] = kernel(np.linalg.norm(data[i] - data, axis=1)/h)/nrmc
        #assert np.sum(S[i, :]) + np.sum(T[i, :]) - 1.0 < tol, "Normalization constant != 1"

    # soft assignment matrix A
    A = np.linalg.pinv(np.identity(n) - T) @ S
    # for given labels from significant clusters it has to hold:
    #assert sum([(np.argmax(A[i, :]) == labels[i]) for i in range(n)]) == n, "Soft assignment failed!"

    return A

def connectivity(A, labels, i, j):
    indices = np.where(labels == i)[0]
    sum_i = np.sum(A[indices, j]) / len(indices)

    indices = np.where(labels == j)[0]
    sum_j = np.sum(A[indices, i]) / len(indices)

    return (sum_i + sum_j)/2

def denoising_threshold(data):
    ''' Remove modes with clusters sizes < n_0. '''
    n = data.shape[0]
    d = data.shape[1]
    n_0 = (n*np.log(n)/20)**(d/(d + 6))
    return n_0

def connectivity_threshold(modes):
    return 1 / (2 * len(modes))

def reconstruct(data, h=None, n=None, c=None):
    ## mode seeking
    if h != None:
        h_0 = h
    else:
        h_0 = kernel_bandwidth(data)
    ms = MeanShift(bandwidth=h_0).fit(data)
    labels = ms.labels_
    modes = ms.cluster_centers_
    k = len(np.unique(labels))

    ## denoising to reduce the number of local modes by thresholding the cluster sizes
    n_0 = 1 / (2 * k)
    sizes = np.array([np.sum(labels == i) for i in range(k)])
    indices = np.where(sizes > n_0)[0]
    k_0 = len(indices)

    ## graph reconstruction by soft cluster assignment and connectivity metric
    A = soft_assignment(data, modes, h_0)
    c_0 = np.mean(A.flatten())
    G = np.zeros((k_0, k_0))
    if n != None:
        n_0 = n
    if c != None:
        c_0 = c
    for i in range(k_0 - 1):
        for j in range(i + 1, k_0):
            label_i = indices[i]
            label_j = indices[j]
            if connectivity(A, labels, label_i, label_j) > c_0:
                G[i, j] = 1
    return labels, modes, indices, G

def visualize_graph(data, labels, modes, indices, G):
    k = len(indices)
    colors = 10*['r.','g.', 'b.', 'c.','m.','y.']

    for i in range(data.shape[0]):
        plt.plot(data[i][0], data[i][1],
                 colors[labels[i]], markersize=10, alpha=0.3)

    for i in range(k):
        label_i = indices[i]
        plt.plot(modes[label_i, 0], modes[label_i, 1],
                 marker='+', color='k', markersize=10)

    for i in range(k - 1):
        for j in range(i + 1, k):
            if G[i, j] == 1:
                label_i = indices[i]
                label_j = indices[j]
                x1, y1 = modes[label_i]
                x2, y2 = modes[label_j]
                plt.plot([x1, x2], [y1, y2], color='k')
    plt.show()

def simple_test():
    # simulate_data()
    data = load_data("test_data")
    labels_sim = load_data("test_labels")

    h_0 = kernel_bandwidth_sklearn(data)
    ms = MeanShift(bandwidth=h_0).fit(data)
    labels = ms.labels_
    modes = ms.cluster_centers_
    k = len(np.unique(labels))
    labels_sim = [int(lab) for lab in labels]
    # visualize_sim(data, labels_sim)

    n_0 = denoising_threshold(data)
    c_0 = connectivity_threshold(modes)
    labels, modes, indices, G = reconstruct(data, h=h_0, n=n_0, c=c_0)
    visualize_graph(data, labels, modes, indices, G)

def claw_test():
    data = load_data("claw/claw_dense_sample")
    visualize(data)

    h = kernel_bandwidth(data)
    ms = MeanShift(bandwidth=h).fit(data)
    labels = ms.labels_
    modes = ms.cluster_centers_
    k = len(np.unique(labels))
    labels_sim = [int(lab) for lab in labels]
    visualize_sim(data, labels_sim)

    labels, modes, indices, G = reconstruct(data)
    visualize_graph(data, labels, modes, indices, G)

if __name__ == "__main__":
    simple_test()
    claw_test()
