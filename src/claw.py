#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" claw.py - Class to visualize the worst case metric graph
embedded in 2-dimensional Euclidean space that is K_1,3 or Claw
glued together in such a way it is hard to distinguish two edges
because they are too close in the embedding space; see (Lecci et al., 2014)
    https://jmlr.csail.mit.edu/papers/volume15/lecci14a/lecci14a.pdf
chapter 3.1.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from random import sample
import os

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 300

class Claw:
    ''' Class for plotting the worst-case embedded graph K_1,3 or Claw
    that is glued together from 3 curves.

        * 4 boundary points [a, (x), t1, t2];
        * n points from the line segment ax without the boundary;
        * (pi/4 * tau) * n points from each circle arc e1 = xt1, e2 = xt2;

    where:

        * n is the number of points on each curve, n>>1;
        * we increase the number of points from circle arcs proportionally to the arc length;
        * tau is the radius of each circle;

    '''
    def __init__(self, title='', n=1000):
        self.tau = np.sqrt(2)
        self.n = n
        self.n1 = self.n + 2 # plus 2, because we remove the boundary points
        # otherwise we are adding the boundary points more than once
        self.n2 = int(np.floor(np.pi/4 * self.tau * self.n)) + 2

    def plot(self, plot_construction=False):
        # plotting parameters
        self.width = 10 # dimensions in inches
        self.height = 10 # dimensions in inches
        self.llc = (-3, -1) # lower left corner
        self.urc = (1, 1) # upper right corner

        self.fig = plt.figure()
        self.fig.set_size_inches(self.width, self.height)
        self.ax = self.fig.add_subplot(111, aspect='equal')

        if plot_construction:
            self.ax.set_xlabel(r'$x$', fontsize=22)
            self.ax.set_ylabel(r'$y$', fontsize=22)

            self.ax.arrow(self.llc[0], 0,
                          self.urc[0] - self.llc[0] - 0.1, 0,
                          head_width=0.05, head_length=0.1,
                          color='grey', fc='grey', ec='grey')

            self.ax.arrow(0, self.llc[1], 0,
                          self.urc[1] - self.llc[1] - 0.1,
                          head_width=0.05, head_length=0.1,
                          color='grey', fc='grey', ec='grey')

            self.ax.set_xticks(range(self.llc[0], self.urc[0]))
            self.ax.set_yticks(range(self.llc[1], self.urc[1]))

            self.ax.grid(True)

            self.ax.set_xlim([self.llc[0], self.urc[0]])
            self.ax.set_ylim([self.llc[1], self.urc[1]])

            ## plot the segment ax in dotted grey
            self.ax.plot([-2, -1], [0, 0], linestyle='--', color='grey')

            ## plot the circles in dotted grey
            g, f = 0, -1 # center of the first circle
            theta = np.linspace(0, 2*np.pi, 100)
            xs = g + self.tau * np.cos(theta)
            ys = f + self.tau * np.sin(theta)
            self.ax.plot(xs, ys, linestyle='--', color='grey')

            # 0, 1 is the center of the second circle
            self.ax.plot(xs, ys + 2, linestyle='--', color='grey')
        else:
            self.ax.set_axis_off()
            ## plot the segment ax in dotted grey
            self.ax.plot([-2, -1], [0, 0], linestyle='--', color='grey')

            ## plot only the arcs in dotted grey
            theta = np.linspace(np.pi/2, np.pi/2 + np.pi/4, self.n2)
            g, f = 0, -1 # center of the first circle
            xs = g + self.tau * np.cos(theta)
            ys = f + self.tau * np.sin(theta)
            self.ax.plot(xs, ys, linestyle='--', color='grey')

            theta = np.linspace(3*np.pi/2, 3*np.pi/2 - np.pi/4, self.n2)
            g, f = 0, 1 # center of the second circle
            xs = g + self.tau * np.cos(theta)
            ys = f + self.tau * np.sin(theta)
            self.ax.plot(xs, ys, linestyle='--', color='grey')

    def boundary_points(self):
        ''' Returns the boundary points a, (x), t1, t2. '''
        a = (-2, 0)
        x = (-1, 0)
        t1 = (0, np.sqrt(2) - 1)
        t2 = (0, 1 - np.sqrt(2))

        return [a, x, t1, t2]

    def segment_ax_points(self):
        ''' Returns n1 points from the segment (ax). '''
        xs = np.linspace(-2, -1, self.n1)
        ys = np.zeros(self.n1)

        # remove the boundary points
        xs = xs[1:(self.n1 - 1)]
        ys = ys[1:(self.n1 - 1)]

        return list(zip(list(xs), list(ys)))

    def arc_e1_points(self):
        ''' Returns n2 points from the circle arc `e1` of length pi/4 * sqrt(2). '''
        theta = np.linspace(np.pi/2, np.pi/2 + np.pi/4, self.n2)
        g, f = 0, -1 # center of the first circle
        xs = g + self.tau * np.cos(theta)
        ys = f + self.tau * np.sin(theta)

        # remove the boundary points
        xs = xs[1:(self.n2 - 1)]
        ys = ys[1:(self.n2 - 1)]

        return list(zip(list(xs), list(ys)))

    def arc_e2_points(self):
        ''' Returns n2 points from the circle arc `e2` of length pi/4 * sqrt(2). '''
        theta = np.linspace(3*np.pi/2, 3*np.pi/2 - np.pi/4, self.n2)
        g, f = 0, 1 # center of the second circle
        xs = g + self.tau * np.cos(theta)
        ys = f + self.tau * np.sin(theta)

        # remove the boundary points
        xs = xs[1:(self.n2 - 1)]
        ys = ys[1:(self.n2 - 1)]

        return list(zip(list(xs), list(ys)))

    def points(self):
        ''' Returns the points from embedded graph K_1,3. '''
        boundary = self.boundary_points()
        segment_ax = self.segment_ax_points()
        arc_e1 = self.arc_e1_points()
        arc_e2 = self.arc_e2_points()

        return boundary + segment_ax + arc_e1 + arc_e2

    def plot_parts(self, plot_construction=False, savefig=False):
        ''' Plots the parts of embedded graph K_1,3 construction. '''
        self.plot(plot_construction)
        self.ax.set_title('Parts of embedded graph $K_{1,3}$ construction')

        boundary = self.boundary_points()
        boundary_np = np.array(boundary)
        self.ax.scatter(boundary_np[:, 0], boundary_np[:, 1], color='black', s=50)

        segment_ax = self.segment_ax_points()
        segment_ax_np = np.array(segment_ax)
        self.ax.scatter(segment_ax_np[:, 0], segment_ax_np[:, 1], color='blue', s=50)

        arc_e1 = self.arc_e1_points()
        arc_e1_np = np.array(arc_e1)
        self.ax.scatter(arc_e1_np[:, 0], arc_e1_np[:, 1], color='green', s=50)

        arc_e2 = self.arc_e2_points()
        arc_e2_np = np.array(arc_e2)
        self.ax.scatter(arc_e2_np[:, 0], arc_e2_np[:, 1], color='red', s=50)

        if savefig:
            self.fig.savefig('../figures/claw_parts.png', dpi=300)
            os.system('convert ../figures/claw_parts.png -trim ../figures/claw_parts.png')

    def plot_claw(self, savefig=False):
        self.plot()
        self.ax.set_title('Embedded graph $K_{1,3}$')

        points = np.array(self.points())
        self.ax.scatter(points[:, 0], points[:, 1], marker='.', color='black')

        if savefig:
            self.fig.savefig('../figures/claw.png', dpi=300)
            os.system('convert ../figures/claw.png -trim ../figures/claw.png')

    def get_sample_points(self, nn):
        points = self.points()
        return sample(points, nn)

    def plot_sample(self, sample_points, sigma=0, savefig=False):
        self.plot()

        dense_sample = self.is_dense(sample_points, sigma)
        if dense_sample:
            self.ax.set_title('Dense sample of points from embedded graph $K_{1,3}$')
        else:
            self.ax.set_title('Not dense sample of points from embedded graph $K_{1,3}$')

        sample_points_np = np.array(sample_points)
        self.ax.scatter(sample_points_np[:, 0], sample_points_np[:, 1],
                        facecolors='k', edgecolors='k', alpha=0.9, s=50)

        if sigma > 0:
            for sample_point in sample_points:
                circle = patches.Circle((sample_point[0], sample_point[1]),
                                        sigma,
                                        facecolor=(0.5, 0.5, 0.5),
                                        edgecolor=(0.0, 0.0, 0.0),
                                        alpha=0.1)
                self.ax.add_patch(circle)

        if savefig:
            if dense_sample:
                self.fig.savefig('../figures/claw_dense_sample.png', dpi=300)
                os.system('convert ../figures/claw_dense_sample.png -trim  \
                    ../figures/claw_dense_sample.png')
            else:
                self.fig.savefig('../figures/claw_not_dense_sample.png', dpi=300)
                os.system('convert ../figures/claw_not_dense_sample.png -trim  \
                    ../figures/claw_not_dense_sample.png')

    def is_dense(self, sample_points, sigma):
        points_EG = self.points()
        for point_EG in points_EG:
            if self.not_covered(point_EG, sample_points, sigma):
                return False
        return True

    def not_covered(self, pointEG, sample_points, sigma):
        for sample_point in sample_points:
            # TODO: including the boundary?
            if self.distance(sample_point, pointEG) < sigma:
                return False
        return True

    def distance(self, p1, p2):
        ''' Euclidean distance between points p1, p2. '''
        p1 = np.array([p1[0], p1[1]])
        p2 = np.array([p2[0], p2[1]])
        return np.linalg.norm(p1 - p2)

if __name__ == "__main__":
    np.random.seed(2)

    savefig = True # set false for faster plotting

    claw = Claw(n=10)
    claw.plot_parts(plot_construction=True, savefig=savefig)
    plt.show()

    claw = Claw()
    claw.plot_claw(savefig=savefig)
    plt.show()

    sigma = 0.1
    sample_points = claw.get_sample_points(10)
    claw.plot_sample(sample_points, sigma, savefig=savefig)
    plt.show()

    sample_points = claw.get_sample_points(100)
    claw.plot_sample(sample_points, sigma, savefig=savefig)
    plt.show()
