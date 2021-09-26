#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" metric_graph_reconstruction.py: Implementation of algorithm for
reconstructing the topology of a metric graph that represents intersecting
or branching filamentary paths embedded in d-dimensional space.

TODO: metric should be more implicit so it's easy to switch from Euclidean
to geodesic or distance induced by Rips or alpha complex

Author: Marko Lalovic <marko.lalovic@yahoo.com>
License: MIT License
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d
from queue import PriorityQueue
import os
import json

class Graph:
    def __init__(self, vertices, edges):
        ''' Graph with vertices associated to points in d-dimensional space.
        Edges are represented simply as pairs of vertices.

        e.g:
            graph = Graph(...)

        Args:
            vertices: list of Point objects
            edges: list of [Point, Point] lists
        '''
        self.vertices = vertices
        self.edges = edges

    def __str__(self):
        vertices = ['v' + str(i+1) for i in range(self.n)]
        name_of = dict(zip(self.vertices, vertices))
        out = 'vertices: \n'
        for vertex, vertex_point in zip(vertices, self.vertices):
            out += '  ' + vertex + ': ' + str(vertex_point) + '\n'

        edges = ['e' + str(i+1) for i in range(self.m)]
        out += '\nedges: \n'
        for edge, edge_points in zip(edges, self.edges):
            out += '  ' + edge + ': ' \
                + ', '.join([name_of[edge_points[0]], name_of[edge_points[1]]]) \
                + '\n'

        out += '\ncomponents: \n'
        for i, cmpt in self.components.items():
            out += '  c' + str(i+1) + ': '
            out += ', '.join([name_of[vertex_point] for vertex_point in cmpt]) \
            + '\n'

        return out

    @property
    def d(self):
        ''' Dimension of embedding space. '''
        return self.vertices[0].d

    @property
    def n(self):
        ''' Number of vertices. '''
        return len(self.vertices)

    @property
    def m(self):
        ''' Number of edges. '''
        return len(self.edges)

    @property
    def k(self):
        ''' Number of connected components.'''
        return len(self.components)

    @property
    def components(self):
        ''' Returns the connected components as a dictionary:
            {i: [Points of component i]}
        '''
        cmpts = []
        visited = []
        for v in self.vertices:
            if not v in visited:
                comp_of_v = self.component(v)
                # add vertices from component to visited
                for u in comp_of_v:
                    visited.append(u)
                cmpts.append(comp_of_v)

        return dict(zip(range(len(cmpts)), cmpts))

    def neighbors(self, v):
        ''' Neighbors of vertex v. '''
        nbrs = []
        for edge in self.edges:
            u1, u2 = edge
            if u1.equals(v):
                nbrs.append(u2)
            elif u2.equals(v):
                nbrs.append(u1)
        return nbrs

    def component(self, v):
        ''' Connected component of v. '''
        def cmpt(v, T):
            nhbs = list(set(self.neighbors(v)) - set(T))
            if nhbs == []:
                return [v]
            else:
                T += nhbs # expand the tree
                for nhb in nhbs:
                    T += cmpt(nhb, T) # expand the tree in BFS way
            return list(set(T))
        return cmpt(v, [v]) # start with T = [v]

    def graph_distance(self, p1, p2):
        ''' Graph distance between points p1, p2. '''
        vertices = [i for i in range(self.n)]
        lenghts = self.graph_distances(p1)
        name_of = dict(zip(self.vertices, vertices))
        return lenghts[name_of[p2]]

    def graph_distances(self, start):
        ''' To compute shortest distances from start to all other vertices. '''
        vertices = [i for i in range(self.n)]
        distances = {v:float('inf') for v in vertices}
        name_of = dict(zip(self.vertices, vertices))
        start = name_of[start]
        distances[start] = 0

        lengths = [[-1 for i in range(self.m)] for j in range(self.m)]
        for edge in self.edges:
            u, v = name_of[edge[0]], name_of[edge[1]]
            weight = 1 # TODO: set weight based on edge lenght
            lengths[u][v] = weight
            lengths[v][u] = weight

        queue = PriorityQueue()
        queue.put((0, start))

        visited = []
        while not queue.empty():
            _, current_vertex = queue.get()
            visited.append(current_vertex)

            for vertex in vertices:
                distance = lengths[current_vertex][vertex]
                if distance != -1:
                    if vertex not in visited:
                        old_length = distances[vertex]
                        new_length = distances[current_vertex] + distance
                        if new_length < old_length:
                            queue.put((new_length, vertex))
                            distances[vertex] = new_length
        return distances

    def show(self):
        ''' Plots the graph structure as a rectilinear drawing.

        TODO: if dimension is more than 3, project the graph, so that the
        associated points can be of any dimension.

        TODO: pass the projection mapping to plot_graph or even curves that are
        associated with the edges.
        '''
        space = Space(self.d)
        space.plot_graph(self)
        space.show()

class Point:
    ''' Supporting class for storing coordinates and labels of points.

    e.g:
        point = Point(...)

    Args:
        coords::tuple(float)
            The coordinates of a point. Should be a tuple of floats.
        label::str
            Should be: 'E' for edge point and 'V' for vertex point.
    '''
    def __init__(self, coords=(), label='P'):
        self.coords = coords
        self.label = label

    def __str__(self):
        return self.label + str(self.coords)

    @property
    def d(self):
        ''' Dimension of embedding space. '''
        return len(self.coords)

    def equals(self, p, eps=1e-4):
        ''' Returns true if point is close to p. '''
        if self.d != p.d:
            return False

        def distance(p1, p2):
            p1 = np.array(p1.coords)
            p2 = np.array(p2.coords)
            return np.linalg.norm(p1 - p2)

        return distance(self, p) < eps

class PointCloud:
    ''' PointCloud Class to hold a list of Point objects.

    e.g:
        point_cloud = PointCloud(...)

    TODO: should be general to work with any dimension of embedding space
    or any distance we are using for reconstructing the graph.

    Test and show capabilities:

        * on Heawood graph embedded on a torus in 3D space
        * on hypercube embedded in 4D space
        * on the earthquake data using geodesic distance
    '''
    def __init__(self, points):
        if points == [] or isinstance(points[0], Point):
            self.points = points
        else:
            raise ValueError('Points must be a list of Point objects.')

    @property
    def vertex_points(self):
        vertex_points = []
        for point in self.points:
            if point.label == 'V':
                vertex_points.append(point)
        return vertex_points

    @property
    def edge_points(self):
        edge_points = []
        for point in self.points:
            if point.label == 'E':
                edge_points.append(point)
        return edge_points

    def __str__(self):
        return '[' + ', '.join([str(point) for point in self.points]) + ']'

    def __len__(self):
        return len(self.points)

    def distance(self, p1, p2):
        ''' Euclidean distance between two points.
        TODO: generalize, so we can use geodesic distance or
        distance induced by Rips-Vietoris graph.
        '''
        p1 = np.array(p1.coords)
        p2 = np.array(p2.coords)
        return np.linalg.norm(p1 - p2)

    def set_distance(self, points1, points2):
        ''' Computes minimum distance between given sets of points points1 and points2. '''
        distances = []
        for point1 in points1:
            for point2 in points2:
                distances.append(self.distance(point1, point2))
        return np.min(np.array(distances))

    def set_center(self, points):
        ''' Computes the center of mass of the given set of points.'''
        points_np = np.array([point.coords for point in points])
        return Point( tuple(np.mean(points_np, axis=0)) )

    def get_shell_points(self, y, radius, delta):
        ''' Returns a list of points between radius and radius + delta around point y.'''
        shell_points = []
        for point in self.points:
            dst = self.distance(y, point)
            if dst >= radius and dst <= radius + delta:
                shell_points.append(point)
        return shell_points

    def rips_vietoris_graph(self, points, delta):
        ''' Constructs the Rips-Vietoris graph on points of parameter delta. '''
        n = len(points)
        vertices = []
        edges = []
        for i in range(n):
            p1 = points[i]
            vertices.append(p1)
            for j in range(i, n):
                p2 = points[j]
                if not p1.equals(p2) and self.distance(p1, p2) < delta:
                    edges.append([p1, p2])
        return Graph(vertices, edges)

    def label_points(self, r, delta):
        ''' Labels the points as edge or vertex points. '''
        for y in self.points:
            shell_points = self.get_shell_points(y, r, delta)
            rips_embedded = self.rips_vietoris_graph(shell_points, delta)
            if rips_embedded.k == 2:
                y.label = 'E'
            else:
                y.label = 'V'

    def get_ball_points(self, center, radius):
        ball_points = []
        for point in self.points:
            dist = self.distance(center, point)
            if dist < radius:
                ball_points.append(point)
        return ball_points

    def expand_vertices(self, p11):
        ''' Re-labels all the points withing distance p11 from
        preliminary vertex points as vertices. '''
        for vertex_point in self.vertex_points:
            ball_points = self.get_ball_points(vertex_point, p11)
            for ball_point in ball_points:
                ball_point.label = 'V'

    def reconstruct(self, delta):
        ''' Reconstructs the graph structure. '''
        # compute the connected components of Rips-Vietoris graphs:
        # Rips_delta(vertex_points), Rips_delta(edge_points)
        rips_V = self.rips_vietoris_graph(self.vertex_points, delta)
        rips_E = self.rips_vietoris_graph(self.edge_points, delta)
        cmpts_V = rips_V.components
        cmpts_E = rips_E.components

        # connected components of Rips_delta(vertex_points) are vertices of
        # reconstructed embedded graph hatG
        # represented here by centers of mass of point clouds
        vertices = []
        for cmpt_V in cmpts_V.values():
            vertices.append(self.set_center(cmpt_V))

        # there is an edge between vertices of hatG if their corresponding
        # connected components in Rips_delta(vertex_points) contain points
        # at distance less than delta from the same component of
        # Rips_delta(edge_points)
        n = len(vertices)
        edges = []
        for i in range(n):
            # we cannot detect loops by setting range(i, n)
            # then each vertex would have a loop
            for j in range(i+1, n):
                for cmpt_E in cmpts_E.values():
                    if self.set_distance(cmpts_V[i], cmpt_E) < delta and \
                       self.set_distance(cmpts_V[j], cmpt_E) < delta:
                        edges.append([vertices[i], vertices[j]])

        return Graph(vertices, edges)

class Space:
    ''' Space on which we plot the graphics.

    e.g:
        space = Space()

    '''
    def __init__(self, dimension,
        figsize=8, remove_ticks=True, label_axes=False):
        if dimension in [2, 3]:
            self.dim = dimension
        else:
            raise ValueError(
                "Space on which we plot the graphics can be 2 or 3 dimensional.")

        plt.rcParams['figure.figsize'] = [figsize, figsize]
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'

        self.fig = plt.figure()
        self.font_size = 28

        if self.dim == 2:
            self.ax = self.fig.add_subplot(111, aspect='equal')

            if label_axes:
                self.ax.set_xlabel('x')
                self.ax.set_ylabel('y')

            if remove_ticks:
                self.ax.set_xticks([])
                self.ax.set_yticks([])

            # set axis colors
            self.ax.spines['top'].set_color('grey')
            self.ax.spines['right'].set_color('grey')
        else:
            self.ax = plt.axes(projection='3d')

            # set aspect ratio
            self.ax.set_box_aspect(aspect = (1,1,1))

            # set viewing angle
            self.ax.azim = 30
            self.ax.dist = 10
            self.ax.elev = 30

            # remove fill
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False

            if remove_ticks:
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.ax.set_zticks([])

            # if remove_axes: self.ax.set_axis_off()

            # set z-axis on the left
            tmp_planes = self.ax.zaxis._PLANES
            self.ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                                     tmp_planes[0], tmp_planes[1],
                                     tmp_planes[4], tmp_planes[5])
            # set axes colors
            self.ax.xaxis.pane.set_edgecolor('gray')
            self.ax.yaxis.pane.set_edgecolor('gray')
            self.ax.zaxis.pane.set_edgecolor('gray')

            if label_axes:
                self.ax.set_xlabel('x')
                self.ax.set_ylabel('y')
                self.ax.set_zlabel('z')

    def show(self, figure_path=''):
        """ Show the space, displaying any graphics on it."""
        if self.dim == 3:
            # fix the aspect ratio for 3d plot; source:
            #https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
            extents = np.array([getattr(self.ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:,1] - extents[:,0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(self.ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        if figure_path != '':
            self.fig.savefig(figure_path, dpi=300)
            os.system('convert ' + figure_path + ' -trim ' + figure_path)

        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def color(self, point, default='green'):
        ''' Returns the color of a point based on its label. '''
        if point.label == 'V':
            return 'red'
        elif point.label == 'E':
            return 'blue'
        else:
            return default

    def plot_point(self, point, color='black', **kwargs):
        ''' Plots a point. '''
        if self.dim == 2:
            x, y = point.coords
            self.ax.scatter(x, y, color=color, s=50, **kwargs)
        else:
            x, y, z = point.coords
            self.ax.scatter3D(x, y, z, color=color, **kwargs)

    def plot_points(self, points, **kwargs):
        coords = np.array([point.coords for point in points])
        colors = list(map(self.color, points))
        if self.dim == 2:
            self.ax.scatter(coords[:, 0], coords[:, 1], color=colors, **kwargs)
        else:
            self.ax.scatter3D(coords[:, 0], coords[:, 1], coords[:, 2],
                              color=colors, depthshade=True, **kwargs)

    def plot_shell(self, center, radius, delta, color='black', **kwargs):
        ''' Plots B(center, radius) and B(center, radius + delta). '''
        if self.dim == 2:
            self.plot_ball(center, radius + delta, color='grey', alpha=0.1)
            self.plot_ball(center, radius + delta, fill=False, alpha=1)
            self.plot_ball(center, radius, color='white', alpha=1)
        else:
            self.plot_ball(center, radius + delta)
            self.plot_ball(center, radius)

    def plot_ball(self, center, radius, color='black', **kwargs):
        """ Plots a ball B(center, radius). """
        if self.dim == 2:
            circle = patches.Circle(center.coords,
                                    radius,
                                    facecolor=color,
                                    edgecolor='k',
                                    linestyle='--',
                                    linewidth='2.2',
                                    zorder=0,
                                    **kwargs)
            self.ax.add_patch(circle)
        else:
            # TODO: simplify for faster drawing
            x, y, z = center.coords
            color='grey'
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, 2* np.pi, 100)
            x += radius * np.outer(np.cos(u), np.sin(v))
            y += radius * np.outer(np.sin(u), np.sin(v))
            z += radius * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.scatter(x, y, z, c=color, marker='o', alpha=0.01*radius)

    def plot_edge(self, p1, p2, color='blue', **kwargs):
        ''' Plots line segment between points p1 and p2. '''
        if self.dim == 2:
            x1, y1 = p1.coords
            x2, y2 = p2.coords
            self.ax.plot([x1, x2], [y1, y2],
                color=color, lw=3.3, **kwargs)
        else:
            x1, y1, z1 = p1.coords
            x2, y2, z2 = p2.coords
            self.ax.plot([x1, x2], [y1, y2], [z1, z2],
                color=color, lw=3.3, **kwargs)

    def plot_graph(self, graph, color='purple', **kwargs):
        ''' Draw the graph as a rectilinear drawing in embedding space. '''
        self.plot_points(graph.vertices)
        for edge in graph.edges:
            self.plot_edge(edge[0], edge[1], color=color)


class MetricGraph:
    '''
    Usage:
        * to draw a metric graph by gluing together a bunch of curves
        * and compute its geometric @properties:
            * lengths of edges -> shortest edge
            e.g.:
                loop1: 18.56
                loop2: 18.56
                -------------
                shortest edge: 18.56

            * smallest curvatures of edges -> local reach
            e.g.:
                loop1: 2.1
                loop2: 2.1
                -----------
                local reach: 2.1

            * angles around vertices -> smallest angle between curves
            e.g.:
                e: 106.98, 73.02
                -----------
                alpha: 73.02

            * TODO: global reach

        * TODO: add getting a dense sample of points from G;
        * TODO: test the reconstruction on it.

    Args: 
    
        To provide a description of metric graph, we use a dictionary, e.g.:
            desc = {
                    'name': 'butterfly',
                    'points': {
                        'a': [0, 0],
                        'b': [20, 0],
                        'c': [20, 15],
                        'd': [0, 15],
                        'e': [10, 7.5]},
                    'vertices': ['e'],
                    'edges': {
                        'loop1': ['e', 'a', 'd', 'e'],
                        'loop2': ['e', 'b', 'c', 'e']}
            }
            
        Or a path to a JSON file with this content, e.g.:
            file = '../data/metric-graphs/butterfly.json'
        
        Points and names should be unique, no duplicates.
         
    '''
    def __init__(self, desc={}, file=''):
        if not desc and not file:
            raise ValueError('Provide a description or a file.')
        if file != '':
            desc = self.load(file)

        self.name = desc['name']
        self.points = [tuple(point) for point in desc['points'].values()]
        names = [name for name in desc['points'].keys()]
        d_points = dict(zip(names, self.points))
        self.name_of = dict(zip(self.points, names))
        self.edge_names = list(desc['edges'].keys())
        vertices = [d_points[vertex] for vertex in desc['vertices']]
        self.vertices = vertices
        edges = []
        for edge in desc['edges'].values():
            edges.append([ d_points[edge[i]] for i in range(len(edge)) ])
        self.edges = edges

    def __str__(self):
        out = 'name: ' + self.name + '\n'
        out += '\npoints: \n'
        for point in self.points:
            out += '  ' + self.name_of[point] + ': ' + str(point) + '\n'
        out += '\nvertices: ' + ', '.join([
            self.name_of[vertex] for vertex in self.vertices]) + '\n'
        out += '\nedges: \n'
        for edge_name, edge in zip(self.edge_names, self.edges):
            control_points = ', '.join([
                self.name_of[control_point] for control_point in edge])
            out += '  ' + edge_name + ': ' + control_points + '\n'
            
        # TODO: print the geometric properties too
        return out

    @property
    def edge_lenghts(self):
        ''' Lenghts of edges of metric graph as a dictionary:
            {edge_name: lenght}
        '''
        lenghts = {}
        for edge_name, edge in zip(self.edge_names, self.edges):
            lenghts[edge_name] = self.edge_lenght(edge)
        return lenghts

    @property
    def shortest_edge(self):
        ''' Returns the shortest edge in the metric graph as:
            (edge_name, lenght) '''
        lenghts = self.edge_lenghts
        return min(lenghts.items(), key=lambda x: x[1])

    @property
    def angles(self, in_degrees=True):
        ''' Returns angles around each vertex enclosed by edges (curves)
        that meet at that vertex. '''
        angles = {}
        for vertex in self.vertices:
            angles[self.name_of[vertex]] = self.get_angles(vertex, in_degrees)
        return angles

    @property
    def smallest_angle(self, in_degrees=True):
        ''' Returns smallest angle between edges in the metric graph. '''
        min_angles = {}
        for vertex, angles in self.angles.items():
            min_angles[vertex] = np.min(angles)
        return min(min_angles.items(), key=lambda x: x[1])

    @property
    def edge_radii(self, nn=500):
        ''' Returns "1/curvature" of the edge = the minimum radius r 
        of a circle touching the edge for each edge. '''
        radii = {}
        for edge_name, edge in zip(self.edge_names, self.edges):
            radii[edge_name] = self.edge_radius(edge, nn=500)
        return radii

    @property
    def local_reach(self):
        ''' Returns the local reach of the metric graph. This is the minimum
        edge_radius over all edges. Where edge radius is 1/curvature of the edge,
        in other words, the minimum radius r of a circle touching the edge. '''
        radii = self.edge_radii
        return min(radii.items(), key=lambda x: x[1])

    def edge_radius(self, edge, nn=500):
        ''' Returns the "1/curvature" of the edge = the minimum radius r of a
        circle touching the edge. '''
        ts = np.linspace(0, 1, nn)
        rs = []
        for t in ts:
            rs.append(self.r_bezier(t, edge))
        return np.min(rs)

    def get_angles(self, vertex, in_degrees=True):
        ''' Returns angles between edges (curves) that meet at the vertex in the metric graph. '''
        ds = []
        for t in [0, 1]:
            for i, edge in enumerate(self.edges):
                d = self.d_bezier(t, edge)
                d /= np.linalg.norm(d)
                ds.append(list(d))
                ds.append(list(-d))
        ds = np.unique(ds, axis=0)

        if in_degrees:
            conversion = (180/np.pi)
        else:
            conversion = 1

        angles = []
        for i in range(ds.shape[0]):
            for j in range(i+1, ds.shape[0]):
                angles.append( np.arccos(np.dot(ds[i], ds[j]))*conversion )
        return list(set(angles))

    def edge_lenght(self, edge, nn=500):
        ''' Computes approximate lenght of an edge. '''
        points = self.bezier_points(edge, nn)
        d = 0
        for i in range(nn - 1):
            d += self.distance(points[i], points[i + 1])
        return d

    def partial_edge_lenghts(self, edge, nn=500):
        ''' Returns approximate lenghts of an edge for a range of
        values of parameter t. '''
        points = self.bezier_points(edge, nn)
        d = 0
        ds = [d]
        for i in range(nn - 1):
            d += self.distance(points[i], points[i + 1])
            ds.append(d)
        return ds

    def n_edge_points(self, density=1):
        ''' Normalizes the lenghts and transforms the normalized
        lenghts to numbers of points on the curves. '''
        lenghts = [self.edge_lenght(edge) for edge in self.edges]

        # normalize the lenghts
        lenghts = np.array(lenghts)
        lenghts /= np.sum(lenghts)

        # transform them to number of edge points
        lenghts *= (density*100)

        return [int(np.round(lenght, 2)) for lenght in lenghts]

    def p_bezier(self, t, edge):
        ''' Returns a point on Bezier curve.
            t: parameter between 0 and 1
        '''
        b = np.array(edge)

        if b.shape[0] == 5:
            return list(t**4 * b[0] + 4*t**3 * (1 - t) * b[1] \
                        + 6*t**2 * (1 - t)**2 * b[2] \
                        + 4*t * (1 - t)**3 * b[3] + (1 - t)**4 * b[4])
        elif b.shape[0] == 4:
            return list((1 - t)**3 * b[0] + 3 * (1 - t)**2 * t * b[1] \
                        + 3 * (1 - t) * t**2 * b[2] \
                        + t**3 * b[3])
        elif b.shape[0] == 2:
            return list( (1 - t) * b[0] +  t * b[1] )
        else:
            raise NotImplementedError

    def d_bezier(self, t, edge):
        ''' Returns derivatives [x', y'] on t at a point on Bezier curve.
            t: parameter between 0 and 1
        '''
        b = np.array(edge)

        if b.shape[0] == 5:
            return list(  4*b[0]*t**3 \
                        - 4*b[1]*t**3 \
                        + 12*b[1]*t**2*(1 - t) \
                        + 6*b[2]*t**2*(2*t - 2) \
                        + 12*b[2]*t*(1 - t)**2 \
                        - 12*b[3]*t*(1 - t)**2 \
                        + 4*b[3]*(1 - t)**3 \
                        - 4*b[4]*(1 - t)**3)
        elif b.shape[0] == 4:
            return list(- 3*b[0]*(1 - t)**2 \
                        + 3*b[1]*t*(2*t - 2) \
                        + 3*b[1]*(1 - t)**2 \
                        - 3*b[2]*t**2 \
                        + 2*b[2]*t*(3 - 3*t) \
                        + 3*b[3]*t**2)
        elif b.shape[0] == 2:
            return list( b[0] - b[1] )
        else:
            raise NotImplementedError

    def dd_bezier(self, t, edge):
        ''' Returns second order derivatives [x'', y''] at a point on Bezier curve.
            t: parameter between 0 and 1
        '''
        b = np.array(edge)

        if b.shape[0] == 5:
            return list(  12*b[0]*t**2 - 24*b[1]*t**2 + 24*b[1]*t*(1 - t) \
                        + 12*b[2]*t**2 + 24*b[2]*t*(2*t - 2) + 12*b[2]*(1 - t)**2 \
                        - 12*b[3]*t*(2*t - 2) - 24*b[3]*(1 - t)**2 + 12*b[4]*(1 - t)**2)
        elif b.shape[0] == 4:
            return list(- 3*b[0]*(2*t - 2) + 6*b[1]*t + 6*b[1]*(2*t - 2) \
                        - 12*b[2]*t + 2*b[2]*(3 - 3*t) + 6*b[3]*t)
        elif b.shape[0] == 2:
            return [0]
        else:
            raise NotImplementedError

    def r_bezier(self, t, edge, eps=1e-4):
        ''' Returns the radius of curvature r(t) at a point on Bezier curve
            kappa = (x'y'' - x''y') / (x'^2 + y'^2)^(3/2)
            r(t) = 1/kappa
        TODO: add curvature for a curve in 3d.
        '''
        d = self.d_bezier(t, edge)
        dd = self.dd_bezier(t, edge)
        kappa_nom = d[0] * dd[1] - dd[0] * d[1]
        kappa_den = (d[0]**2 + d[1]**2)**(3/2)
        if kappa_den < eps:
            return 0
        else:
            return np.abs(kappa_den/kappa_nom)

    def n_bezier(self, t, edge):
        ''' Returns the normal vector at a point on Bezier curve.
        TODO: add for a curve in 3d.
        '''
        d = self.d_bezier(t, edge)
        d /= np.linalg.norm(d)
        return [-d[1], d[0]]

    def edge_sigma_tube(self, edge, n, sigma):
        ''' Returns the points on sigma tube around the edge
        of the metric graph.'''
        b_points = self.regular_bezier_points(edge, n)
        n_points = self.normal_bezier_points(edge, n)

        b_points = np.array(b_points)
        n_points = np.array(n_points)
        n_points *= sigma

        s_tube1 = b_points + n_points
        s_tube2 = b_points - n_points

        s_tube = list(s_tube1) + list(s_tube2)
        s_tube = [list(point) for point in s_tube]
        return s_tube

    def bezier_points(self, edge, n):
        points = []
        ts = np.linspace(0, 1, n)
        for t in ts:
            points.append(self.p_bezier(t, edge))
        return points

    def regular_bezier_points(self, edge, n):
        points = []
        ts = self.regular_ts(edge, n)
        for t in ts:
            points.append(self.p_bezier(t, edge))
        return points

    def normal_bezier_points(self, edge, n):
        points = []
        ts = self.regular_ts(edge, n)
        for t in ts:
            points.append(self.n_bezier(t, edge))
        return points

    def regular_ts(self, edge, n=50, nn=500):
        ''' Returns ts that cut up the curve at regular intervals. '''
        ts = np.linspace(0, 1, nn)
        ds = self.partial_edge_lenghts(edge, nn)
        ds = np.array(ds)
        ds /= ds[-1]

        targets = list(np.linspace(0, 1, n))[1:-1]
        foundts = [0]
        for target in targets:
            for i in range(len(ds) - 1):
                if target > ds[i] and target < ds[i + 1]:
                    foundts.append( (ts[i] + ts[i+1])/2 )
        foundts += [1]
        return foundts

    def save(self, file_name):
        with open(file_name, 'w') as file:
            json.dump(self.d, file, indent=4)

    def load(self, file_name):
        with open(file_name) as file:
            d = json.load(file)
        return d

    def distance(self, p1, p2):
        ''' Euclidean distance between two points. '''
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)

    def get_points(self, density=1):
        ''' Returns regularly spaced points from metric graph. '''
        points = []
        for vertex in self.vertices:
            points.append(vertex)

        ns = self.n_edge_points(density)
        for edge, n in zip(self.edges, ns):
            points += self.regular_bezier_points(edge, n)
        return points

    def get_st_points(self, density=1, sigma=0.5):
        ''' Returns regularly spaced points on the boundary
        of sigma tube around the metric graph. '''
        points_st = [] # points on the sigma tube
        ns = self.n_edge_points(density)
        for edge, n in zip(self.edges, ns):
            points_st += self.edge_sigma_tube(edge, n, sigma)
        return points_st

    def show(self, density=4, sigma_tube=True, marker='.'):
        ''' Plots the metric graph drawing. '''
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        points = self.get_points(density)
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1],
            color='black', s=10, marker=marker)
        if sigma_tube:
            points_st = self.get_st_points(density)
            points_st = np.array(points_st)
            ax.scatter(points_st[:, 0], points_st[:, 1],
                color='red', s=10, marker=marker)
        plt.show()

# some helper functions
def geogebra_point_cloud(point_cloud):
    ''' Some help for exporting point clouds to GeoGebra, e.g.
        >>> geogebra_point_cloud(PointCloud([Point(1, 2), Point(3, 4)]))
        Execute[{"P0 = (1, 2)", "P1 = (3, 4)"}]
    '''
    out = 'Execute[{'
    labels = ['P' + str(i) for i in range(len(point_cloud))]
    for label, point in zip(labels, point_cloud.points):
        x, y = point.coords
        out += '"' + label + ' = (' + str(x) + ', ' + str(y) + ')", '

    out = out[:-2]
    out += '}]'
    print(out)

def geogebra_bezier_curve(curve_name, control_points):
    ''' Some help for drawing Bezier curves in GeoGebra, e.g.
            geogebra_bezier_curve('quartic', ['A', 'B', 'C', 'D'])
    '''
    b = control_points
    if len(b) == 4: # quartic
        out = curve_name + ' = Curve['
        out += '(1 - t)^3 x(' + b[0] + ') + 3 (1 - t)^2 t x(' + b[1] \
            + ') + 3 (1 - t) t^2 x(' + b[2] + ') + t^3 x(' + b[3] + '),'
        out += '(1 - t)^3 y(' + b[0] + ') + 3 (1 - t)^2 t y(' + b[1] \
            + ') + 3 (1 - t) t^2 y(' + b[2] + ') + t^3 y(' + b[3] + '),'
        out += ' t, 0, 1]'
        print(out)
    else: # quintic
        out = curve_name + ' = Curve['
        out += 't^4 x(' + b[0] + ') + 4t^3 (1 - t) x(' + b[1] \
            + ') + 6t^2 (1 - t)^2 x(' + b[2] + ') + 4t (1 - t)^3 x(' \
            + b[3] + ') + (1 - t)^4 x(' + b[4] + '),'
        out += 't^4 y(' + b[0] + ') + 4t^3 (1 - t) y(' + b[1] \
            + ') + 6t^2 (1 - t)^2 y(' + b[2] + ') + 4t (1 - t)^3 y(' \
            + b[3] + ') + (1 - t)^4 y(' + b[4] + '),'
        out += ' t, 0, 1]'

def save_sample(sample_points, file_name):
    np.savetxt('../data/samples/' + file_name + '.out',
               np.array(sample_points),
               delimiter=',')

def load_sample(file_name):
    sample_points_np = np.loadtxt('../data/samples/' + file_name + '.out',
                      delimiter=',')
    return [tuple(pt) for pt in sample_points_np]
