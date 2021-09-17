#!/usr/bin/python3
# -*- coding: utf-8 -*-
''' control-points.py - For Bezier curves that represent parts of a metric graph. We get a dog by gluing the curves together.
'''

A = (8.53, 17.15)
A1 = (20.71, 16.84)
B = (10.63, 17.24)
B1 = (15.6, 18.08)
C = (10.8, 17.82)
C1 = (18.98, 13.33)
D = (12.39, 14.64)
D1 = (19.2, 13.67)
E = (11.93, 12.9)
E1 = (16.95, 11.83)
F = (8.89, 8.95)
F1 = (18.01, 10.14)
G = (10.98, 8.77)
G1 = (19.34, 12.24)
H = (18.84, 14.27)
H1 = (20.52, 9.93)
I = (16.12, 15.56)
I1 = (9.95, 11.53)
J = (19.04, 12.74)
J1 = (11.99, 11.39)
K = (15.72, 9.22)
K1 = (18.64, 18.9)
L = (19.65, 8.59)
M = (9.36, 16.99)
N = (9.72, 17.43)
O = (11.73, 16.5)
P = (11.39, 15.38)
Q = (12.33, 13.45)
R = (11.97, 14.22)
S = (10.89, 12.45)
T = (9.89, 9.83)
U = (12.26, 12.21)
V = (13.24, 9.72)
W = (14.35, 14.24)
Z = (15.77, 14.85)

parts = {
    'head': [A, M, N, B],
    'ear': [B, C],
    'neck': [B, O, P, D],
    'back': [D, W, Z, H],
    'front': [D, R, Q, E],
    'ass': [H, D1, C1, J],
    'leg1': [E, S, I1, T, F],
    'leg2': [E, U, J1, V, G],
    'leg3': [J, E1, F1, K],
    'leg4': [J, G1, H1, L],
    'tail': [H, A1, K1, B1, I]
}
