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
import math
import copy

CANVAS_WIDTH = 10
CANVAS_HEIGHT = 10
MAX = 28 # max x,y coordinate
MIN = -2 # min x,y coordinate

class EmbeddedGraph:
    def __init__(self, nodes, edges):
        ''' Graph with points embedded in the plane.'''
        self.nodes = PointCloud(nodes)
        self.edges = [PointCloud(edge) for edge in edges]

    def __str__(self):
        points = [str(point) for point in self.nodes.points]
        edges = [str(edge) for edge in self.edges]
        components = [str(cmpt_emb_G) for cmpt_emb_G in self.components.values()]

        return "nodes: {}\nedges: {}\ncomponents: {}".format(
            str(points), str(edges),  str(components))

    @property
    def n(self):
        ''' Number of nodes in EmbeddedGraph.'''
        return len(self.nodes.points)

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
            point_of[i] = self.nodes.points[i]

        for i, cmpt_G in cmpts_G.items():
            cmpts_emb_G[i] = PointCloud( [point_of[j] for j in cmpt_G] )

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
        nx.draw(graph_G, pos, font_size=10,
                node_color='red', with_labels=True)
        plt.show()


class Point:
    ''' Class Point for storing coordinates and label of a point.

    Args:
        x::float
            The x coordinate of the point.
        y::float
            The y coordinate of the point.
        label::str
            Should be: 'E' for edge point and 'V' for vertex point.
    '''
    def __init__(self, x=0, y=0, label=''):
        self.x = x
        self.y = y
        if label not in ('E', 'V', ''):
            raise ValueError ("Label must be 'E' or 'V'")
        self.label = label

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.label)

    def equal(self, p):
        return (self.x == p.x) and (self.y == p.y)

class PointCloud:
    def __init__(self, points):
        ''' PointCloud Class to hold a list of Point objects.'''
        if points == [] or isinstance(points[0], Point):
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

        return Point(x, y)

    def __str__(self):
        return '[' + ','.join(['{!s}'.format(p) for p in self.points]) + ']'

    def __len__(self):
        return len(self.points)

    def contains(self, p):
        for pt in self.points:
            if pt.x == p.x and pt.y == p.y:
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


class Canvas:
    """ Class Canvas on which we draw the graphics."""
    def __init__(self, title, xlabel='X', ylabel='Y',
                 p1=Point(MIN, MIN), p2=Point(MAX, MAX)):
        self.fig = plt.figure()
        self.fig.set_size_inches(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.ax = self.fig.add_subplot(111, aspect='equal')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(p1.x, p2.x))
        plt.yticks(range(p1.y, p2.y))
        self.ax.grid(True)
        self.ax.set_xlim([p1.x, p2.x])
        self.ax.set_ylim([p1.y, p2.y])

    def show(self):
        """ Show the canvas, displaying any graphics drawn on it."""
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()


def draw_point(canvas, pt, radius=0.25, color='blue', **kwargs):
    ''' Draws a point.'''
    point = patches.Circle((pt.x, pt.y),
                        radius=radius,
                        fill=True,
                        facecolor=color,
                        **kwargs)
    canvas.ax.add_patch(point)

def draw_points(canvas, points):
    for point in points:
        if point.label == 'V':
            color = 'red'
        elif point.label == 'E':
            color = 'blue'
        else:
            color = 'green'
        draw_point(canvas, point, color=color)

def draw_graph(canvas, emb_G, color='black'):
    for pt in emb_G.nodes.points:
        draw_point(canvas, pt, color=color)

    for edge in emb_G.edges:
        draw_edge(canvas, edge.points[0], edge.points[1], color=color)

    plt.show()

def draw_ball(canvas, pt, radius=5, color='blue', **kwargs):
    """ Draws a ball."""
    # draw_point(canvas, pt, radius=0.2, color=color)
    circle = patches.Circle((pt.x, pt.y),
                        radius,
                        fill=False,
                        edgecolor=color,
                        linestyle='dotted',
                        linewidth='2.2',
                        **kwargs)
    canvas.ax.add_patch(circle)

def draw_edge(canvas, p1, p2, color='blue', **kwargs):
    """ Draws a line segment between points p1 and p2."""
    line = patches.FancyArrow(p1.x, p1.y,
                              p2.x - p1.x,
                              p2.y - p1.y,
                              color=color,
                              linewidth='3.3',
                              **kwargs)
    canvas.ax.add_patch(line)

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
        point_of[i] = emb_G.nodes.points[i]

    number_of = {}
    for i in range(emb_G.n):
        number_of[emb_G.nodes.points[i]] = i

    nodes = list(point_of.keys())
    edges = []
    for i in range(emb_G.n):
        for j in range(i + 1, emb_G.n):
            # test if there is an edge between Points v1 and v2
            v1 = emb_G.nodes.points[i]
            v2 = emb_G.nodes.points[j]

            for edge in emb_G.edges:
                u1 = edge.points[0]
                u2 = edge.points[1]
                if v1.equal(u1) and v2.equal(u2) or \
                   v1.equal(u2) and v2.equal(u1):
                    edges.append( (number_of[v1], number_of[v2]) )

    return Graph(nodes, edges)

def distance(p1, p2):
    ''' Euclidean distance between p1, p2.'''
    # TODO: generalize if p \in R^3
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

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

def reconstruct(point_cloud, delta=3, r=2, p11=1.5, show=False):
    ''' Implementation of Aanjaneya's metric graph reconstruction algorithm.'''
    ## label the points as edge or vertex points
    for center in point_cloud.points:
        shell_points = get_shell_points(point_cloud.points, center, r, delta)
        rips_embedded = rips_vietoris_graph(delta, shell_points)

        if rips_embedded.k == 2:
            center.label = 'E'
        else:
            center.label = 'V'
    if show:
        canvas = Canvas('After labeling')
        draw_points(canvas, point_cloud.points)

    # re-label all the points withing distance p11 from vertex points as vertices
    for center in point_cloud.vertex_points:
        ball_points = get_ball_points(point_cloud.edge_points, center, p11)
        for ball_point in ball_points:
            ball_point.label = 'V'
    if show:
        canvas = Canvas('After re-labeling')
        draw_points(canvas, point_cloud.points)

    # reconstruct the graph structure
    # compute the connected components of Rips-Vietoris graphs:
    # R_delta(vertex_points), R_delta(edge_points)
    rips_V = rips_vietoris_graph(delta, point_cloud.vertex_points)
    rips_E = rips_vietoris_graph(delta, point_cloud.edge_points)
    cmpts_V = rips_V.components
    cmpts_E = rips_E.components

    nodes_emb_E = []
    for i, cmpt_E in cmpts_E.items():
        nodes_emb_E.append(cmpt_E.center)
    emb_E = EmbeddedGraph(nodes_emb_E, [])

    nodes_emb_G = []
    for i, cmpt_V in cmpts_V.items():
        nodes_emb_G.append(cmpt_V.center)

    n = len(nodes_emb_G)
    edges_emb_G = []
    for i in range(n):
        for j in range(i + 1, n):
            for cmpt_E in cmpts_E.values():
                if cmpts_V[i].distance(cmpt_E) < delta and \
                   cmpts_V[j].distance(cmpt_E) < delta:
                    edges_emb_G.append([nodes_emb_G[i], nodes_emb_G[j]])

    emb_G = EmbeddedGraph(nodes_emb_G, edges_emb_G)
    if show:
        canvas = Canvas('Result')
        draw_points(canvas, point_cloud.points)
        draw_graph(canvas, emb_G, color='red')
        draw_graph(canvas, emb_E, color='black')
        print(emb_E)

    return emb_G

def draw_labeling(point_cloud, delta=3, r=2, p11=1.5, step=0):
    ''' Draw the labeling step of the algorithm.'''

    canvas = Canvas('Labeling points as edge or vertex points')
    draw_points(canvas, point_cloud.points)

    if step == 0:
        step = int(np.floor(len(point_cloud.points)/4)) - 2
    center = point_cloud.points[step]

    draw_ball(canvas, center, r, 'black')
    draw_ball(canvas, center, r + delta, color='black')

    shell_points = get_shell_points(point_cloud.points, center, r, delta)
    rips_embedded = rips_vietoris_graph(delta, shell_points)

    draw_graph(canvas, rips_embedded, color='red')

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

    canvas = Canvas('Re-labeling points as vertex points')
    draw_points(canvas, point_cloud.points)

    i = int(np.floor(len(point_cloud.points)/4)) - 2
    center = point_cloud.points[i]

    draw_ball(canvas, center, radius=p11, color='black')

    ball_points = get_ball_points(point_cloud.edge_points, center, p11)
    for ball_point in ball_points:
        draw_point(canvas, ball_point, color='green')

    plt.show()
