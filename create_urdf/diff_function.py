import pybullet as p
import time
import pybullet_data
from math import sqrt, cos, sin, pi
import os
import numpy as np


def set_random_angle():
    initial_q1 = np.random.random_sample(size=5) * 2 - 1
    initial_q1 = initial_q1 * np.pi / 3

    initial_q2 = np.random.random_sample(size=5) * 2 - 1
    initial_q2 = initial_q2 * np.pi / 3

    initial_q3 = np.array([initial_q2[0], initial_q2[1],
                           initial_q2[2], initial_q2[3], initial_q2[4]])
    initial_q4 = np.array([initial_q1[0], initial_q1[1],
                           initial_q1[2], initial_q1[3], initial_q1[4]])

    initial_q11 = np.append(initial_q1, initial_q2)
    initial_q22 = np.append(initial_q3, initial_q4)

    initial_q = np.append(initial_q11, initial_q22)
    # print(initial_q)

    initial_q[0] = float(0)
    initial_q[4] = float(0)

    initial_q[5] = float(0)
    initial_q[9] = float(0)

    initial_q[10] = float(0)
    initial_q[14] = float(0)

    initial_q[15] = float(0)
    initial_q[19] = float(0)

    return initial_q


def sign(x1, y1, x2, y2, x3, y3):

    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)


def isInside(x, y, x1, y1, x2, y2, x3, y3):

    d1 = sign(x, y, x1, y1, x2, y2)
    d2 = sign(x, y, x2, y2, x3, y3)
    d3 = sign(x, y, x3, y3, x1, y1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not(has_neg and has_pos)


def tri_leg_balance(leg1, leg2, leg3, leg4, base):
    flag = 0
    check1 = isInside(base[0], base[1], leg1[0], leg1[1],
                      leg2[0], leg2[1], leg3[0], leg3[1])
    check2 = isInside(base[0], base[1], leg1[0], leg1[1],
                      leg2[0], leg2[1], leg4[0], leg4[1])
    check3 = isInside(base[0], base[1], leg1[0], leg1[1],
                      leg3[0], leg3[1], leg4[0], leg4[1])
    check4 = isInside(base[0], base[1], leg2[0], leg2[1],
                      leg3[0], leg3[1], leg4[0], leg4[1])

    if check1 is True:
        flag += 1
    if check2 is True:
        flag += 1
    if check3 is True:
        flag += 1
    if check4 is True:
        flag += 1

    if flag >= 2:
        return True
    else:
        return False


def get_uv(euler):
    alfa = euler[0]

    beta = euler[1]

    gamma = euler[2]

    rot_x = [[1, 0, 0],
             [0, np.cos(alfa), -np.sin(alfa)],
             [0, np.sin(alfa), np.cos(alfa)]]

    rot_y = [[np.cos(beta), 0, np.sin(beta)],
             [0, 1, 0],
             [-np.sin(beta), 0, np.cos(beta)]]

    rot_z = [[np.cos(gamma), -np.sin(gamma), 0],
             [np.sin(gamma), np.cos(gamma), 0],
             [0, 0, 1]]

    y_link = [0, 1, 0]

    rot_z = np.asarray(rot_z)
    rot_y = np.asarray(rot_y)
    rot_x = np.asarray(rot_x)
    y_link = np.asarray(y_link)
    R1 = np.dot(rot_z, rot_y)
    R2 = np.dot(R1, rot_x)
    vect = np.dot(R2, y_link)
    return vect
