#!/usr/bin/python3
# -*- coding: utf-8 -*-
""" metric_graph_reconstruction.py: Implementation of algorithm for
reconstructing the topology of a metric graph that represents intersecting
or branching filamentary paths embedded in 2 dimensional space.

Author: Marko Lalovic <marko.lalovic@yahoo.com>
License: MIT License
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import axes3d
import copy

class EmbeddedGraph:
    def __init__(self, nodes, edges):
        ''' Graph with points embedded in the plane.'''
        self.nodes = nodes
        self.edges = edges

    def __str__(self):
        points = [str(point) for point in self.nodes]
        edges = [[str(edge[0]), str(edge[1])] for edge in self.edges]
        components = [[str(point) for point in cmpt_emb_G]
            for cmpt_emb_G in self.components.values()]

        return "nodes: {}\nedges: {}\ncomponents: {}".format(
            str(points), str(edges),  str(components))

    @property
    def n(self):
        ''' Number of nodes in EmbeddedGraph.'''
        return len(self.nodes)

    @property
    def m(self):
        ''' Number of edges in EmbeddedGraph.'''
        return len(self.edges)

    @property
    def k(self):
        ''' Number of connected components of EmbeddedGraph.'''
        return len(self.components)

    @property
    def components(self):
        ''' Computes connected components of EmbeddedGraph'''
        graph_G = graph(self)
        cmpts_G = graph_G.components

        cmpts_emb_G = {}
        point_of = {}
        for i in range(self.n):
            point_of[i] = self.nodes[i]

        for i, cmpt_G in cmpts_G.items():
            cmpts_emb_G[i] = [point_of[j] for j in cmpt_G]

        return cmpts_emb_G

class Graph:
    def __init__(self, nodes, edges):
        ''' Graph represented with nodes and edges.'''
        if isinstance(nodes, list):
            self.nodes = nodes
        else:
            self.nodes = list(nodes)
        if isinstance(edges, list):
            self.edges = edges
        else:
            self.edges = list(edges)

    @property
    def n(self):
        ''' Returns the number of nodes. '''
        return len(self.nodes)

    @property
    def m(self):
        ''' Returns the number of edges. '''
        return len(self.edges)

    @property
    def k(self):
        ''' Returns the number of connected components. '''
        return len(self.components)

    @property
    def components(self):
        ''' Returns the connected components. '''
        cmpts = {}
        k = 0
        unvisited = copy.copy(self.nodes)
        for v in self.nodes:
            if v in unvisited:
                comp_of_v = component(v, self.nodes, self.edges)
                # remove visited nodes in component from unvisited
                unvisited = list(set(unvisited) - set(comp_of_v))
                cmpts[k] = comp_of_v
                k += 1

        return cmpts

    def __str__(self):
        return "nodes: {}\nedges: {}".format(str(self.nodes), str(self.edges))

    def draw(self):
        ''' Draws the graph. '''
        graph_G = nx.Graph()
        graph_G.add_nodes_from(self.nodes)
        graph_G.add_edges_from(self.edges)

        pos = nx.spring_layout(graph_G)
        nx.draw(graph_G, pos, font_size=50,
                node_color='black', with_labels=False)

        plt.show()


class Point:
    ''' Class Point for storing coordinates and label of a point.

    Args:
        x::float
            The x coordinate of the point.
        y::float
            The y coordinate of the point.
        z::float
            The z coordinate of the point.
        label::str
            Should be: 'E' for edge point and 'V' for vertex point.
    '''
    def __init__(self, x=0, y=0, z=0, label=''):
        self.x = x
        self.y = y
        self.z = z
        if label not in ('E', 'V', ''):
            raise ValueError ("Label must be 'E' or 'V'")
        self.label = label

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.z, self.label)

    def equal(self, p):
        return (self.x == p.x) and (self.y == p.y) and (self.z == p.z)

class PointCloud:
    def __init__(self, points):
        ''' PointCloud Class to hold a list of Point objects.'''
        if points == [] or isinstance(points[0], Point):
            # shift to non-negative values for 3d effect
            shifted = []
            for point in points:
                shifted.append([point.x, point.y, point.z])

            shifted = np.array(shifted)
            if np.any(shifted[:, 2]): # if 3d
                shifted = np.array(shifted)
                minimums = np.amin(shifted, axis=0)
                shifted = shifted + 2 * np.abs(minimums)
                points = [Point(point[0], point[1], point[2]) for point in shifted]

            self.points = points
        else:
            raise ValueError("Args must be a list of Points.")

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

    @property
    def center(self):
        ''' Center of mass of the point cloud.'''
        x = np.mean(np.array( [point.x for point in self.points] ))
        y = np.mean(np.array( [point.y for point in self.points] ))
        z = np.mean(np.array( [point.z for point in self.points] ))

        return Point(x, y)

    def __str__(self):
        return '[' + ','.join(['{!s}'.format(p) for p in self.points]) + ']'

    def __len__(self):
        return len(self.points)

    def contains(self, p):
        for pt in self.points:
            if pt.x == p.x and pt.y == p.y and pt.z == p.z:
                return True
        return False

    def append(self, p):
        self.points.append(p)

    def difference(self, pl):
        difference = PointCloud([])
        for pt in self.points:
            if not pl.contains(pt):
                difference.append(pt)
        return difference

    def distance(self, point_cloud):
        ''' Computes minimum distance from self to another point cloud.'''
        distances = []
        for p1 in self.points:
            for p2 in point_cloud.points:
                distances.append(distance(p1, p2))

        return np.min(np.array(distances))

class Space:
    ''' Space on which we draw the graphics. '''
    def __init__(self, dimension):
        if dimension in [2, 3]:
            self.dim = dimension
        else:
            raise ValueError("Dimension should be an integer 2 or 3.")

        plt.rcParams['figure.figsize'] = [20, 15]
        plt.rcParams['text.usetex'] = True
        plt.rcParams.update({'font.size': 18})
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'

        self.fig = plt.figure()
        self.font_size = 28

        if self.dim == 2:
            self.ax = self.fig.add_subplot(111, aspect='equal')

            # # label axes
            # self.ax.set_xlabel(r'$x$', fontsize=self.font_size)
            # self.ax.set_ylabel(r'$y$', fontsize=self.font_size)

            # remove ticks
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

            # remove axes ticks
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.ax.set_zticks([])

            # self.ax.set_axis_off()

            # set z-axis on the left
            tmp_planes = self.ax.zaxis._PLANES
            self.ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                                     tmp_planes[0], tmp_planes[1],
                                     tmp_planes[4], tmp_planes[5])

            # remove fill
            self.ax.xaxis.pane.fill = False
            self.ax.yaxis.pane.fill = False
            self.ax.zaxis.pane.fill = False

            # set axes colors
            self.ax.xaxis.pane.set_edgecolor('gray')
            self.ax.yaxis.pane.set_edgecolor('gray')
            self.ax.zaxis.pane.set_edgecolor('gray')

            # label axes
            self.ax.set_xlabel(r'$x$', fontsize=self.font_size)
            self.ax.set_ylabel(r'$y$', fontsize=self.font_size)
            self.ax.set_zlabel(r'$z$', fontsize=self.font_size)

    def show(self):
        """ Show the space, displaying any graphics drawn on it."""
        if self.dim == 3:
            # fix the aspect ratio for 3d plot
            # source: https://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
            extents = np.array([getattr(self.ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
            sz = extents[:,1] - extents[:,0]
            centers = np.mean(extents, axis=1)
            maxsize = max(abs(sz))
            r = maxsize/2
            for ctr, dim in zip(centers, 'xyz'):
                getattr(self.ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

        plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def draw_point(self, point, color='black', **kwargs):
        ''' Draws a point. '''
        if self.dim == 2:
            self.ax.scatter(point.x, point.y, color=color, s=50)
        else:
            self.ax.scatter3D(point.x, point.y, point.z, color=color, s=50)

    def draw_points(self, points):
        if self.dim == 2:
            for point in points:
                if point.label == 'V':
                    color = 'red'
                elif point.label == 'E':
                    color = 'blue'
                else:
                    color = 'black'
                self.draw_point(point, color=color)
        else:
            # set size of scatter markers to give the appearance of depth
            sizes = [point.x + point.y/2 + 2*point.z for point in points]
            a = np.min(sizes)
            b = np.max(sizes)
            sizes = (sizes - a) / (b - a)
            c = 30
            d = 60
            sizes = c + sizes * (d - c)

            # to speed it up, we add all the points to the plot at the same time
            points_np = []
            for point in points:
                points_np.append([point.x, point.y, point.z])
            points_np = np.array(points_np)

            # TODO: set colors based on labels
            # for now, let all be black
            colors = ['black']*100
            self.ax.scatter3D(points_np[:, 0], points_np[:, 1], points_np[:, 2],
                              s=sizes, color=colors, depthshade=True)


    def draw_shell(self, center, radius=5, color='black', **kwargs):
        """ Draws a ball around center. """
        if self.dim == 2:
            circle = patches.Circle((center.x, center.y),
                                    radius,
                                    fill=False,
                                    edgecolor=color,
                                    linestyle='dotted',
                                    linewidth='2.2',
                                    **kwargs)
            self.ax.add_patch(circle)
        else:
            color='grey'
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, 2* np.pi, 100)
            x = center.x + radius * np.outer(np.cos(u), np.sin(v))
            y = center.y + radius * np.outer(np.sin(u), np.sin(v))
            z = center.z + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.scatter(x, y, z, c=color, marker='o', alpha=0.01*radius)
            self.ax.plot_surface(x, y, z, antialiased=True,
                color=color, rstride=1, cstride=1, alpha=0.05*radius)
            self.ax.plot_wireframe(x, y, z, color=color, alpha=0.02*radius)

    def draw_graph(self, emb_G, color='black'):
        for point in emb_G.nodes:
            self.draw_point(point, color=color)

        for edge in emb_G.edges:
            self.draw_edge(edge[0], edge[1], color=color)

        plt.show()

    def draw_edge(self, p1, p2, color='blue', **kwargs):
        """ Draws a line segment between points p1 and p2."""
        line = patches.FancyArrow(p1.x, p1.y,
                                  p2.x - p1.x,
                                  p2.y - p1.y,
                                  color=color,
                                  linewidth='3.3',
                                  **kwargs)
        self.ax.add_patch(line)

def nhbs(v, graph_G):
    ''' Returns neighbors of v in graph G. '''
    neighbors = []
    for edge in graph_G.edges:
        u1, u2 = edge
        if u1 == v:
            neighbors.append(u2)
        elif u2 == v:
            neighbors.append(u1)
    return neighbors

def component(v, nodes, edges):
    ''' Wrapper of comp.'''
    G = Graph(nodes, edges)
    return comp(v, G, [v]) # T=[v] at the start

def comp(v, graph_G, T):
    N = list(set(nhbs(v, graph_G)) - set(T))
    if N == []:
        return [v]
    else:
        T += N # expand the tree
        for n in N:
            T += comp(n, graph_G, T) # expand the tree (BFS)
    return list(set(T))
# tests
# graph_G = nx.petersen_graph()
# graph_G = Graph(list(graph_G.nodes()), list(graph_G.edges()))
# graph_G.components[0] == graph_G.nodes # True
# graph_G = Graph(list(range(10)), [])
# len(graph_G.components) == 10 # True


def graph(emb_G):
    ''' Transform from EmbeddedGraph to Graph.'''

    point_of = {}
    for i in range(emb_G.n):
        point_of[i] = emb_G.nodes[i]

    number_of = {}
    for i in range(emb_G.n):
        number_of[emb_G.nodes[i]] = i

    nodes = list(point_of.keys())
    edges = []
    for i in range(emb_G.n):
        for j in range(i + 1, emb_G.n):
            # test if there is an edge between Points v1 and v2
            v1 = emb_G.nodes[i]
            v2 = emb_G.nodes[j]

            for edge in emb_G.edges:
                u1 = edge[0]
                u2 = edge[1]
                if v1.equal(u1) and v2.equal(u2) or \
                   v1.equal(u2) and v2.equal(u1):
                    edges.append( (number_of[v1], number_of[v2]) )

    return Graph(nodes, edges)

def distance(p1, p2):
    ''' Euclidean distance between p1, p2.'''
    p1 = np.array([p1.x, p1.y, p1.z])
    p2 = np.array([p2.x, p2.y, p2.z])
    return np.linalg.norm(p1 - p2)

def get_shell_points(points, center, r, delta):
    ''' Returns a list of points between r and r + delta around the center
    point.'''
    shell_points = []
    for point in points:
        d = distance(center, point)
        if d >= r and d <= r + delta:
            shell_points.append(point)

    return shell_points

def get_ball_points(points, center, r):
    ball_points = []
    for point in points:
        d = distance(center, point)
        if d < r:
            ball_points.append(point)

    return ball_points

def rips_vietoris_graph(delta, points):
    ''' Constructs the Rips-Vietoris graph of parameter delta whose nodes
    are points of the shell.'''
    n = len(points)
    nodes = []
    edges = []
    for i in range(n):
        p1 = points[i]
        nodes.append(p1)
        for j in range(i, n):
            p2 = points[j]
            if not p1.equal(p2) and distance(p1, p2) < delta:
                edges.append([p1, p2])

    return EmbeddedGraph(nodes, edges)

def reconstruct(point_cloud, delta=3, r=2, p11=1.5):
    ''' Implementation of Aanjaneya's metric graph reconstruction algorithm.'''
    ## Labeling
    # label the points as edge or vertex points
    for center in point_cloud.points:
        shell_points = get_shell_points(point_cloud.points, center, r, delta)
        rips_embedded = rips_vietoris_graph(delta, shell_points)

        if rips_embedded.k == 2:
            center.label = 'E'
        else:
            center.label = 'V'

    ## Expansion
    # re-label all the points withing distance p11 from preliminary vertex
    # points as vertices
    for center in point_cloud.vertex_points:
        ball_points = get_ball_points(point_cloud.edge_points, center, p11)
        for ball_point in ball_points:
            ball_point.label = 'V'

    # Reconstructing the graph structure
    # compute the connected components of Rips-Vietoris graphs:
    # Rips_delta(vertex_points), Rips_delta(edge_points)
    rips_V = rips_vietoris_graph(delta, point_cloud.vertex_points)
    rips_E = rips_vietoris_graph(delta, point_cloud.edge_points)
    cmpts_V = rips_V.components
    cmpts_E = rips_E.components

    # tranform lists of points in components to point clouds
    for i, cmpt_points in cmpts_V.items():
        cmpts_V[i] = PointCloud(cmpt_points)

    for j, cmpt_points in cmpts_E.items():
        cmpts_E[j] = PointCloud(cmpt_points)

    # connected components of Rips_delta(vertex_points) are vertices of
    # reconstructed graph hatG represented by centers of mass of point clouds
    nodes_emb_G = []
    for i, cmpt_V in cmpts_V.items():
        nodes_emb_G.append(cmpt_V.center)

    # there is an edge between vertices of hatG if their corresponding
    # connected components in Rips_delta(vertex_points) contain points
    # at distance less than delta from the same component of
    # Rips_delta(edge_points)
    n = len(nodes_emb_G)
    edges_emb_G = []
    for i in range(n):
        for j in range(i + 1, n):
            for cmpt_E in cmpts_E.values():
                if cmpts_V[i].distance(cmpt_E) < delta and \
                   cmpts_V[j].distance(cmpt_E) < delta:
                    edges_emb_G.append([nodes_emb_G[i], nodes_emb_G[j]])

    emb_G = EmbeddedGraph(nodes_emb_G, edges_emb_G)

    return emb_G

def draw_labeling(point_cloud, delta=3, r=2, p11=1.5, step=0):
    ''' Draw the labeling step of the algorithm.'''

    # labeling points as edge or vertex points
    space = Space(2)
    space.draw_points(point_cloud.points)

    if step == 0:
        step = int(np.floor(len(point_cloud.points)/4)) - 2
    center = point_cloud.points[step]

    space.draw_shell(center, r, 'black')
    space.draw_shell(center, r + delta, color='black')

    shell_points = get_shell_points(point_cloud.points, center, r, delta)
    rips_embedded = rips_vietoris_graph(delta, shell_points)

    space.draw_graph(rips_embedded, color='red')

    plt.show()

def draw_re_labeling(point_cloud, delta=3, r=2, p11=1.5):
    # label points as edge or vertex
    for center in point_cloud.points:
        shell_points = get_shell_points(point_cloud.points, center, r, delta)
        rips_embedded = rips_vietoris_graph(delta, shell_points)

        if rips_embedded.k == 2:
            center.label = 'E'
        else:
            center.label = 'V'

    # re-labeling points as vertex points
    space = Space(2)
    space.draw_points(point_cloud.points)

    i = int(np.floor(len(point_cloud.points)/4)) - 2
    center = point_cloud.points[i]

    space.draw_shell(center, radius=p11, color='black')

    ball_points = get_ball_points(point_cloud.edge_points, center, p11)
    for ball_point in ball_points:
        space.draw_point(ball_point, color='green')

    plt.show()
