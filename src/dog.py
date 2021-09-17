#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" dog.py - Toy example for testing metric graph reconstruction:

    * draw a metric graph G of a dog by gluing together a bunch of Bezier curves;
    * get a dense sample of points from G;
    * test the reconstruction on it.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 300

dog_data = '../data/dog/control-points.py' # loading the control points from GeoGebra
exec(open(dog_data).read())

class Plane:
    def __init__(self):
        self.width = 10
        self.height = 10
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, aspect='equal')
        self.fig.set_size_inches(self.width, self.height)
        self.ax.set_axis_off()

    def show(self):
        plt.show()

    def plot(self, points, color='k'):
        points = np.array(points)
        self.ax.scatter(points[:, 0], points[:, 1], s=5, c=color)

    def plot_sample(self, sample_points, delta, savefig=False):
        lengths = get_lengths()
        for part in parts.keys():
            ts = np.linspace(0, 1, lengths[part])
            points = xy_points(ts, parts[part])
            points_np = np.array(points)
            self.ax.plot(points_np[:, 0], points_np[:, 1], linestyle='--', color='grey')

        sample_points_np = np.array(sample_points)
        self.ax.scatter(sample_points_np[:, 0], sample_points_np[:, 1],
                        facecolors='k', edgecolors='k', alpha=0.9, s=50)

        if delta > 0:
            for sample_point in sample_points:
                self.plot_ball(sample_point, delta/2, color='grey', alpha=0.1)
                self.plot_ball(sample_point, delta/2, fill=False, alpha=0.1)

    def plot_ball(self, center, radius, color='black', **kwargs):
        circle = patches.Circle((center[0], center[1]),
                                radius,
                                facecolor=color,
                                edgecolor='k',
                                linestyle='--',
                                linewidth='2.2',
                                zorder=0,
                                **kwargs)
        self.ax.add_patch(circle)

def bezier(t, part):
    '''Drawing Bezier curves of different orders. '''
    points = np.array(part)

    if points.shape[0] == 5:
        return list(t**4 * points[0] + 4*t**3 * (1 - t) * points[1] \
                    + 6*t**2 * (1 - t)**2 * points[2] \
                    + 4*t * (1 - t)**3 * points[3] + (1 - t)**4 * points[4])
    elif points.shape[0] == 4:
        return list((1 - t)**3 * points[0] + 3 * (1 - t)**2 * t * points[1] \
                    + 3 * (1 - t) * t**2 * points[2] \
                    + t**3 * points[3])
    else:
        return list( t * points[0] + (1 - t) * points[1] )

def derivative(t, part):
    ''' Derivatives of Bezier curves for sigma-tube. '''
    points = np.array(part)

    if points.shape[0] == 5:
        return list(4*points[0]*t**3 - 4*points[1]*t**3 \
                    + 12*points[1]*t**2*(1 - t) \
                    + 6*points[2]*t**2*(2*t - 2) \
                    + 12*points[2]*t*(1 - t)**2 \
                    - 12*points[3]*t*(1 - t)**2 \
                    + 4*points[3]*(1 - t)**3 \
                    - 4*points[4]*(1 - t)**3)
    elif points.shape[0] == 4:
        return list(-3*points[0]*(1 - t)**2 \
                    + 3*points[1]*t*(2*t - 2) \
                    + 3*points[1]*(1 - t)**2 \
                    - 3*points[2]*t**2 \
                    + 2*points[2]*t*(3 - 3*t) \
                    + 3*points[3]*t**2)
    else:
        return list(points[0] - points[1])

def normal(t, part):
    ''' Returns the normal vector. '''
    dxy = derivative(t, part)
    dxy /= np.linalg.norm(dxy)

    return [-dxy[1], dxy[0]]

def normal_points(ts, part, sigma=0.5):
    ''' For sigma-tube. '''
    nns = []
    for t in ts:
        nns.append(normal(t, part))

    xys = xy_points(ts, part)
    ups = list(np.array(xys) + np.array(nns) * sigma)
    downs = list(np.array(xys) - np.array(nns) * sigma)

    return ups, downs

def xy_points(ts, part):
    xys = []
    for t in ts:
        xys.append(bezier(t, part))
    return xys

def get_lengths():
    ''' Returns approximate curve lengths. '''
    lengths = {}
    for part in parts.keys():
        ts = np.linspace(0, 1, 1000)
        xys = xy_points(ts, parts[part])
        s = 0
        for i in range(len(xys) - 1):
            s += np.linalg.norm(np.array(xys[i]) - np.array(xys[i+1]))
        lengths[part] = int(s * 100)

    return lengths #[int(x) for x in (np.array(list(lengths.values())) * 100)]

def get_points():
    points = []
    lengths = get_lengths()
    for part in parts.keys():
        ts = np.linspace(0, 1, lengths[part])
        points += xy_points(ts, parts[part])
    return points

def get_sample_points(n):
    points = get_points()
    return random.sample(points, n)

def geogebra(part, points):
    ''' Some help for drawing Bezier curves in GeoGebra, e.g.
            geogebra('quartic', ['A', 'B', 'C', 'D'])
    '''
    if len(points) == 4:
        out = part + ' = Curve['
        out += '(1 - t)^3 x(' + points[0] + ') + 3 (1 - t)^2 t x(' + points[1] \
            + ') + 3 (1 - t) t^2 x(' + points[2] + ') + t^3 x(' + points[3] + '),'
        out += '(1 - t)^3 y(' + points[0] + ') + 3 (1 - t)^2 t y(' + points[1] \
            + ') + 3 (1 - t) t^2 y(' + points[2] + ') + t^3 y(' + points[3] + '),'
        out += ' t, 0, 1]'
        print(out)
    else: # quintic
        out = part + ' = Curve['
        out += 't^4 x(' + points[0] + ') + 4t^3 (1 - t) x(' + points[1] \
            + ') + 6t^2 (1 - t)^2 x(' + points[2] + ') + 4t (1 - t)^3 x(' \
            + points[3] + ') + (1 - t)^4 x(' + points[4] + '),'
        out += 't^4 y(' + points[0] + ') + 4t^3 (1 - t) y(' + points[1] \
            + ') + 6t^2 (1 - t)^2 y(' + points[2] + ') + 4t (1 - t)^3 y(' \
            + points[3] + ') + (1 - t)^4 y(' + points[4] + '),'
        out += ' t, 0, 1]'

# TODO: join some parts with the claw in metric_graph_reconstruction
def is_dense(sample_points, delta):
    pointsEG = get_points()
    for pointEG in pointsEG:
        if not_covered(pointEG, sample_points, delta):
            return False
    return True

def not_covered(pointEG, sample_points, delta):
    for sample_point in sample_points:
        if distance(sample_point, pointEG) < delta/2:
            return False
    return True

def distance(p1, p2):
    ''' Euclidean distance between points p1, p2. '''
    p1 = np.array([p1[0], p1[1]])
    p2 = np.array([p2[0], p2[1]])
    return np.linalg.norm(p1 - p2)

def save_sample(sample_points, file_name):
    np.savetxt('../data/dog/' + file_name + '.out',
               np.array(sample_points),
               delimiter=',')

def load_sample(file_name):
    sample_points_np = np.loadtxt('../data/dog/' + file_name + '.out',
                      delimiter=',')
    return [tuple(pt) for pt in sample_points_np]
