import os

import cal_single as cals
from random import seed
import random


seed()


def create():
    a0 = random.sample(range(0, 19), 4)

    F1 = [a0[0]]
    F2 = [a0[1]]
    F3 = [a0[2]]
    F4 = [a0[3]]

    for i in range(3):
        value = random.randint(0, 11)
        # print(value)
        F1.append(value)

    for i in range(3):
        value = random.randint(0, 11)
        # print(value)
        F2.append(value)

    for i in range(3):
        value = random.randint(0, 11)
        # print(value)
        F3.append(value)

    for i in range(3):
        value = random.randint(0, 11)
        # print(value)
        F4.append(value)

    #print(F1, F2, F3, F4)

    str = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
        F1[0], F1[1], F1[2], F1[3], F2[0], F2[1], F2[2], F2[3], F3[0], F3[1], F3[2], F3[3], F4[0], F4[1], F4[2], F4[3])

    cals.write_urdf(F1, F2, F3, F4)
    return str


def create_symm_notop(log_path):

    a0 = random.sample([6, 7, 10, 11, 15, 16], 2)
    F1 = [a0[0]]
    F2 = [a0[1]]
    F3 = []
    F4 = []

    F = [F1, F2, F3, F4]

    for i in range(2):
        # if (a0[i]) == (7 or 8 or 6 or 9):
        if a0[i] in [7, 8, 6, 9]:
            gap = a0[i] - 7.5
            symm = 7.5 - gap
            symm = int(symm)
            # print(symm)
            F[3 - i].append(symm)
        elif a0[i] in [11, 13, 10, 14]:
            gap = a0[i] - 12
            symm = 12 - gap
            symm = int(symm)
            # print(symm)
            F[3 - i].append(symm)
        elif a0[i] in [16, 18, 15, 19]:
            gap = a0[i] - 17
            symm = 17 - gap
            symm = int(symm)
            # print(symm)
            F[3 - i].append(symm)

    for i in range(3):
        value = random.randint(0, 11)
        # print(value)
        F1.append(value)

    for i in range(3):
        value = random.randint(0, 11)
        # print(value)
        F2.append(value)

    for i in range(2):
        for j in range(1, 4):
            if j == 4:
                rev = [6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7]
                F[3 - i].append(rev[F[i][j]])
            else:
                if F[i][j] in [0, 6]:
                    F[3 - i].append(F[i][j])

                else:
                    symm = 12 - F[i][j]
                    symm = int(symm)
                    F[3 - i].append(symm)

    robot_name = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
        F1[0], F1[1], F1[2], F1[3], F2[0], F2[1], F2[2], F2[3], F3[0], F3[1], F3[2], F3[3], F4[0], F4[1], F4[2], F4[3])
    save_data_pth =log_path+'/robot_urdf_search/'
    if robot_name in set(os.listdir(save_data_pth)):
        return None
    cals.write_urdf(F1, F2, F3, F4, save_data_pth)
    return robot_name

# for i in range(1):
#    create_symm_notop()
