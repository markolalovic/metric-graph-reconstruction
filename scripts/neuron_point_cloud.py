#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" neuron_point_cloud.py - simple script to visualize the point cloud of the neuron cr22e
from the hippocampus of a rat. Data is from NeuroMorpho.Org (Ascoli et al., 2007):
    http://neuromorpho.org/neuron_info.jsp?neuron_name=cr22e
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import axes3d
import os

plt.rcParams['figure.figsize'] = [20, 15]
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# reading the swc data file, format specs:
# http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
points = []
with open('../data/neuron/cr22e.swc') as f:
    for i, line in enumerate(f):
        data = line.split()
        # data = [i, structure, x, y, z, radius, parent]
        points.append(data[2:5])

points = np.array(points)
points = points.astype(np.float64)
xs, ys, zs = points[:, 0], -points[:, 1], points[:, 2]

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = plt.axes(projection='3d')

# set aspect ratio
ax.set_box_aspect(aspect = (1,1,1))

# and view angle
ax.azim = 30
ax.dist = 10
ax.elev = 30

# remove axes ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# set z-axis on the left
tmp_planes = ax.zaxis._PLANES
ax.zaxis._PLANES = ( tmp_planes[2], tmp_planes[3],
                     tmp_planes[0], tmp_planes[1],
                     tmp_planes[4], tmp_planes[5])

# remove fill
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# set axes colors
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')

# label axes
ax.set_xlabel(r'$x$', fontsize=22)
ax.set_ylabel(r'$y$', fontsize=22)
ax.set_zlabel(r'$z$', fontsize=22)

# draw the point cloud
ax.scatter3D(xs, ys, zs, color='black', s=50, alpha=1)

# and save it to ../figures/neuron.png
fig.savefig('../figures/neuron.png', dpi=300)
os.system('convert ../figures/neuron.png -trim ../figures/neuron.png')

# # save a movie
# for angle in range(0, 360, 1):
#     ax.view_init(elev=30, azim=angle)
#     fig.savefig('../figures/neuron_movie%d.png' % angle)
