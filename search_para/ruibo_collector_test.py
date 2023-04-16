import random
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import step
import numpy as np
from ruibo_envsin import *
from multiprocessing import Process, Manager, cpu_count
import argparse
from datetime import date
import os
from tqdm import tqdm
import math
import shutil
import statistics
from tqdm.contrib.telegram import tqdm, trange

from multiprocessing import Process


def find_best(robot_name, mode, number, num_epoch=600, steps=6):
    robot_info = getRobotURDFAndInitialJointAngleFilePath(
        robot_name, urdf_path=SELECT_PARA_FILE_PATH)

    env = RobotEnv(robot_info, [0, 0, 0.25], follow_robot=False)

    testnum = 1
    env.sleep_time = 0
    max_score = 0
    best_para = [] * testnum
    b_para = []
    count = 0
    re_flag = False
    # env.force = 1.8
    fd_flag = False
    for epoch in range(num_epoch):
        print(robot_name, epoch)

        score_list = []
        avg_score = -999
        fall = False

        if max_score == 0:
            # keep improve the parameter
            try:
                sin_para = np.loadtxt(SELECT_PARA_FILE_PATH + "%s/para_mode%s_diffenv_%s.csv" % (robot_name, mode, number))

            except:
                sin_para = random_para()
        else:
            sin_para = b_para[count]

            count += 1
            # print(count)
            if count == 15:
                re_flag = True

        if re_flag == True:
            b_para = [sin_para] * 10

            for _ in range(4):
                b_para.append(random_para())

            b_para.append(sin_para)
            b_para = np.asarray(b_para)
            b_para = batch_random_para(b_para)
            count = 0
            re_flag = False
            continue
        # print(env.force)
        # env.force = random.uniform(1.3, 1.8)
        # print(env.force)

        for j in range(testnum):
            # fri2 = random.uniform(0.1, 0.8)
            fri2 = 0.99
            energy = 0
            obs = env.reset(friction=fri2)
            # print(env.force)
            delta_z = []
            delta_yaw = []

            for i in range(steps):
                obs, r, done, _, energy = env.step(sin_para)
                delta_z.append(obs[2])
                delta_yaw.append(obs[5])
                # print("yaw is", obs[5])
                if done == True:
                    fall = True
                    break
                poss = env.robot_location()[0]
                if math.atan2(abs((poss[0])), abs(poss[1])) * 180 / np.pi >= 30:
                    fall = True
                    break
                yaww = env.robot_location()[1][2]
                if abs(yaww*180/np.pi) >= 30:
                    fall = True
                    break

            if fall == True:
            # print("The robot fall")
                break

            pos, ori = env.robot_location()
            delta_yaw = np.asarray(delta_yaw)
            delta_z = np.asarray(delta_z)
            score = move_score(pos, ori, delta_z, delta_yaw, energy, mode)

            # print("score", score)
            if score > 0:
                score_list.append(score)
            else:
                break

        if len(score_list) == testnum:
            # print("list len", score_list)
            avg_score = sum(score_list) / len(score_list)
            # print("avg", avg_score)

        if (avg_score > max_score) and (len(score_list) == testnum):
            max_score = avg_score
            # print("Find it")
            # print("the score is", max_score)
            best_para = sin_para
            np.savetxt(SELECT_PARA_FILE_PATH + "%s/para_mode%s_diffenv_%s.csv" %
                       (robot_name, mode, number), np.asarray(best_para))

            forward_distance = env.robot_location()[0][1]
            if forward_distance >= 1:
                print("Find")
                original = SELECT_PARA_FILE_PATH + robot_name
                target = GOOD_DATA_PATH + robot_name
                shutil.copytree(original, target)
                break

            b_para = [sin_para] * 10

            for _ in range(4):
                b_para.append(random_para())

            b_para.append(sin_para)
            b_para = np.asarray(b_para)
            # print("before", b_para)
            b_para = batch_random_para(b_para)
            # print("after", b_para)
            count = 0

        if (epoch == num_epoch-1):
            forward_distance = env.robot_location()[0][1]
            if forward_distance < 1:
                print(robot_name, "fail")
                original = SELECT_PARA_FILE_PATH + robot_name
                target = BAD_DATA_PATH + robot_name
                shutil.copytree(original, target)
                # shutil.rmtree(original)

    return max_score
            # if max_score > 2.2:
            #     max_score = 0
            #     env.force += 0.1


def move_score(pos, ori, delta_z, delta_yaw, energy, mode):
    # 0,1,2,3 : forward, back, left, right
    k_y = [1, -1, 0.95, 0.95]
    k_x = [0.8, 0.8, 0.95, 0.95]
    k_yaw = [0.85, 0.85, 1, -1]
    k_z = 0.75
    total_z = sum(abs(delta_z))
    total_yaw = ori[2]
    steps = 10
    # print("pos", pos[0], pos[1])
    if energy >= 4.6:
        k_e = 1 - (4.6 / 20)
    else:
        k_e = 1

    if mode == 0:
        # print(math.atan2(abs((pos[0])), abs(pos[1])) * 180 / np.pi)
        # if math.atan2(abs((pos[0])), abs(pos[1])) * 180 / np.pi >= 30:
        #     score = -999
        # else:
        score = k_e * (k_y[mode] * pos[1] - k_x[mode] * abs(pos[0]) - k_yaw[mode] * abs(total_yaw) - k_z * total_z)
    elif mode == 1:
        if math.atan2(abs((pos[0])), abs(pos[1])) * 180 / np.pi >= 30:
            score = -999
        else:
            score = k_e * (k_y[mode] * pos[1] - k_x[mode] * abs(pos[0]) -
                           k_yaw[mode] * abs(total_yaw) - k_z * total_z)
    elif mode == 2:
        score = k_e * (k_yaw[mode] * total_yaw - k_y[mode] *
                       abs(pos[1]) - k_x[mode] * abs(pos[0]) - k_z * total_z) / steps
    elif mode == 3:
        score = k_e * (k_yaw[mode] * total_yaw - k_y[mode] *
                       abs(pos[1]) - k_x[mode] * abs(pos[0]) - k_z * total_z) / steps

    return score


def check_2m_para(fri123, robot_name, mode, number, steps=6):
    robot_info = getRobotURDFAndInitialJointAngleFilePath(
        robot_name, urdf_path=SELECT_PARA_FILE_PATH)

    env = RobotEnv(robot_info, [0, 0, 0.25], follow_robot=False)
    # env.force = 1.2905
    # env.force = 1.35
    # env.max_velocity = 2
    # if GUI_flag == False:
    env.sleep_time = 0
    para = np.loadtxt(SELECT_PARA_FILE_PATH +
                      "%s/para_mode%s_diffenv_%s.csv" % (robot_name, mode, number))
    # fri = np.loadtxt(SELECT_PARA_FILE_PATH + "%s/friction_forward_over_2m.csv" % robot_name)
    # env.friction = fri
    total = 0
    obs = env.reset(fri123)
    # time.sleep(2)
    for i in range(steps):
        # cur_obs = np.copy(env.get_obs())
        # print(i,"wow",cur_obs)
        # obs, r, done, _, act, real = env.step2(para)
        obs, r, done, _, a = env.step(para)
        # act_real_Plot(act,real)
        # total += obs[1]
    # pos, ori = env.robot_location()
    # print("z",obs[2])

        # print("obs", obs)
        # max_reward, reward_id = step_reward(obs=obs, ini_state=cur_obs)
    # print(total)

    print(env.robot_location())


def act_real_Plot(act, real):
    x = range(16)
    # fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12) = plt.subplots(12, 1)
    # ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

    #######################################
    data_path = "hardware/real_sense/realworld_data1.csv"

    raw_data = np.loadtxt(data_path)
    print(raw_data.shape)
    cur_state = raw_data[:, :18]
    action = raw_data[:, 18:30]
    next_state = raw_data[:, 30:]
    real_action = next_state[:, 6:]

    #######################################
    plt.figure()
    for i in range(12):
        plt.subplot(4, 3, i + 1)
        plt.plot(x, act[:, i], label='action')
        plt.plot(x, real[:, i], label='sim_Joints')

        plt.plot(x, action[:, i], label='realworld_action')
        plt.plot(x, next_state[:, i], label='realworld__Joints')

        plt.tight_layout()
        plt.title("Joint Number" + str(i + 10))
        plt.legend()

    plt.show()


if __name__ == '__main__':
    SELECT_PARA_FILE_PATH = '../data/robot_warehouse/'
    BAD_DATA_PATH = '../data/bad_config/'
    GOOD_DATA_PATH = '../data/good_config/'
    try:
        os.mkdir(BAD_DATA_PATH)
        os.mkdir(GOOD_DATA_PATH)
    except OSError:
        pass

    GUI_flag = True
    # GUI_flag = False

    if GUI_flag == True:
        physicsClient = p.connect(1)  # p.GUI = 1

    else:
        physicsClient = p.connect(2)  # p.DIRECT = 2

    sc_list = []

    mylist = os.listdir(SELECT_PARA_FILE_PATH)
    random.shuffle(mylist)

    for ID in range(100):
        print(ID)
        robot_name = mylist[ID]
        find_best(robot_name, 0, "test")
        # check_2m_para(0.99, robot_name, 0, "test")
        # time.sleep(0.5)

    # print(robot_name)
    # print(type(robot_name))
    # check_2m_para(a, robot_name, 3, "test")
        # grade = find_best(robot_name, 0, "test")
        # grade = find_best(robot_name, 1, "test")
        # grade = find_best(robot_name, 2, "test")
    #     grade = find_best(robot_name, 3, "test")
    #
    # old 516 + 1 = 517
        # robot_name = standRobot(STAND_ROBOT_PATH, ROBOTID)