import os
import math as m
import random
import time
import pybullet_data as pd
import gym
from search_para.controller import *
import numpy as np
#
# random.seed(2022)
# np.random.seed(2022)


class meta_sm_Env(gym.Env):
    def __init__(self, init_q, robot_camera=False, urdf_path='CAD2URDF'):

        self.stateID = None
        self.robotid = None
        self.v_p = None
        self.q = None
        self.p = None
        # self.last_q = None
        self.last_p = None
        self.obs = [0] * 18
        self.mode = p.POSITION_CONTROL

        self.sleep_time = 0  # decrease the value if it is too slow.

        self.maxVelocity = 1.5  # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.force = 1.8

        # self.max_velocity = 1.8
        # self.force = 1.6

        self.n_sim_steps = 60
        self.motor_action_space = np.pi / 3
        self.urdf_path = urdf_path
        self.camera_capture = False
        self.robot_camera = robot_camera
        self.friction = 0.99
        self.robot_view_path = None
        self.sub_step_num = 16
        self.joint_moving_idx = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]
        self.initial_moving_joints_angle = np.asarray([np.pi / 2 * init_q[idx] for idx in self.joint_moving_idx])

        self.action_space = gym.spaces.Box(low=-np.ones(16, dtype=np.float32), high=np.ones(16, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=-np.ones(18, dtype=np.float32) * np.inf,
                                                high=np.ones(18, dtype=np.float32) * np.inf)

        self.log_obs = []
        self.log_action = []
        self.log_sub_state = []
        self.count = 0

        p.setAdditionalSearchPath(pd.getDataPath())

    def World2Local(self, pos_base, ori_base, pos_new):
        psi, theta, phi = ori_base
        R = np.array([[m.cos(phi), -m.sin(phi), 0],
                      [m.sin(phi), m.cos(phi), 0]])
        R_in = R.T
        pos = np.asarray([pos_base[:2]]).T
        R = np.hstack((R_in, np.dot(-R_in, pos)))
        pos2 = list(pos_new[:2]) + [1]
        vec = np.asarray(pos2).T
        local_coo = np.dot(R, vec)
        return local_coo

    def get_obs(self):
        self.last_p = self.p
        self.last_q = self.q
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        # Change robot world-XY_coordinates to robot-XY_coordinates
        self.v_p = self.World2Local(self.last_p, self.last_q, self.p)
        self.v_p[2] = self.p[2] - self.last_p[2]

        self.v_q = self.q - self.last_q

        # correct the values if over 180 to -180.
        if self.v_q[2] > 1.57:
            self.v_q[2] = self.q[2] - self.last_q[2] - 2 * np.pi
        elif self.v_q[2] < -1.57:
            self.v_q[2] = (2 * np.pi + self.q[2]) - self.last_q[2]

        jointInfo = [p.getJointState(self.robotid, i) for i in self.joint_moving_idx]
        jointVals = np.array([[joint[0]] for joint in jointInfo]).flatten()
        self.obs = np.concatenate([self.v_p, self.v_q, jointVals])

        return self.obs

    def act(self, a):

        for j in range(12):
            pos_value = a[j]
            p.setJointMotorControl2(self.robotid, self.joint_moving_idx[j], controlMode=self.mode,
                                    targetPosition=pos_value,
                                    force=self.force,
                                    maxVelocity=self.maxVelocity)

        for _ in range(self.n_sim_steps):
            p.stepSimulation()
            if Time_SLEEP_FLAG:
                time.sleep(1/960)

            # if self.render:
            #     # Capture Camera
            #     if self.camera_capture == True:
            #         basePos, baseOrn = p.getBasePositionAndOrientation(self.robotid)  # Get model position
            #         basePos_list = [basePos[0], basePos[1], 0.3]
            #         p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
            #                                      cameraTargetPosition=basePos_list)  # fix camera onto model
            # time.sleep(self.sleep_time)
            # self.log_sub_state.append(self.get_obs())

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)

        robotStartPos = [0, 0, 0.3]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robotid = p.loadURDF(self.urdf_path, robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=0)
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=140, cameraPitch=-45, cameraTargetPosition=[0, 0, 0])

        p.changeDynamics(self.robotid, 2, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 5, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 8, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 11, lateralFriction=self.friction)
        p.changeVisualShape(self.robotid, -1, rgbaColor=[118 / 255, 182 / 255, 238 / 255, 1])

        for link in [0, 2,
                     5, 7,
                     10, 12,
                     15, 17]:
            p.changeVisualShape(self.robotid, link, rgbaColor=[2 / 255, 33 / 255, 105 / 255, 1])

        for link in [1, 3, 6, 8, 11, 13, 16, 18]:
            p.changeVisualShape(self.robotid, link, rgbaColor=[118 / 255, 182 / 255, 238 / 255, 1])

        for j in range(12):
            pos_value = self.initial_moving_joints_angle[j]
            p.setJointMotorControl2(self.robotid, self.joint_moving_idx[j], controlMode=self.mode,
                                    targetPosition=pos_value, force=self.force, maxVelocity=100)

        for _ in range(100):
            p.stepSimulation()
        self.stateID = p.saveState()
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)

        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        return self.get_obs()

    def resetBase(self):
        p.restoreState(self.stateID)

        self.last_p = 0
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)

        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        return self.get_obs()

    def sin_move(self, ti, para, Period=16):

        # print(para)
        s_action = np.zeros(12)
        # print(ti)
        s_action[0] = para[0] * np.sin(ti / Period * 2 * np.pi + para[2])  # right   hind
        s_action[3] = para[1] * np.sin(ti / Period * 2 * np.pi + para[3])  # right  front
        s_action[6] = para[1] * np.sin(ti / Period * 2 * np.pi + para[4])  # left  front
        s_action[9] = para[0] * np.sin(ti / Period * 2 * np.pi + para[5])  # left  hind

        s_action[1] = para[6] * np.sin(ti / Period * 2 * np.pi + para[2])  # right   hind
        s_action[4] = para[7] * np.sin(ti / Period * 2 * np.pi + para[3])  # right   front
        s_action[7] = para[7] * np.sin(ti / Period * 2 * np.pi + para[4])  # left  front
        s_action[10] = para[6] * np.sin(ti / Period * 2 * np.pi + para[5])  # left  hind

        s_action[2] = para[8] * np.sin(ti / Period * 2 * np.pi + para[2])  # right   hind
        s_action[5] = para[9] * np.sin(ti / Period * 2 * np.pi + para[3])  # right   front
        s_action[8] = para[9] * np.sin(ti / Period * 2 * np.pi + para[4])  # left  front
        s_action[11] = para[8] * np.sin(ti / Period * 2 * np.pi + para[5])  # left  hind

        return s_action

    def step(self, sin_para):

        r = 0
        penalty = 0
        step_done = False
        # range 1 to step_num +1 so that the robot can achieve the original pos after current action.
        for epoch in range(6):
            if BEHAVIOR != 'f':
                obs = meta_env.resetBase()
            if friction_noise_Flag:
                self.friction = friction_noise_list[epoch%3]
            for a_i in range(1, self.sub_step_num + 1):
                a_i_add = self.sin_move(a_i, sin_para, self.sub_step_num)
                norm_a = self.initial_moving_joints_angle + a_i_add
                norm_a = np.random.normal(norm_a,action_noise)
                norm_a = np.clip(norm_a, -1, 1)
                a = norm_a * self.motor_action_space
                self.act(a)
                pos, ori = self.robot_location()

                if BEHAVIOR == 'f':
                    penalty += abs(ori[0]) + abs(ori[1]) +abs(ori[2]) +abs(pos[0])
                    step_done = self.check()


                else:
                    penalty += abs(pos[0]) + abs(pos[1]) + abs(pos[2]-0.13)
                    step_done = self.check_turn()

                if step_done == True:
                    break


            pos, ori = self.robot_location()

            # FORWARD
            if BEHAVIOR == 'f':
                r += 20 * pos[1] - abs(ori[2]) - 0.5 * abs(pos[0]) -  0.1*penalty

            if BEHAVIOR == 'l':
                r += ori[2] - 1 * penalty

            if BEHAVIOR == 'r':
                r += -ori[2] - 0.1 * penalty

            if step_done:
                obs = meta_env.resetBase()
                r = -100
                break
            else:
                obs = pos

        self.count += 1
        return obs, r, step_done, {}

    def robot_location(self):
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check_turn(self):
        pos, ori = self.robot_location()
        if abs(pos[0]) > 0.1:
            abort_flag = True
        # elif (abs(ori[0]) > np.pi / 6) or (abs(ori[1]) > np.pi / 6) or (abs(ori[2]) > np.pi / 6):
        #     abort_flag = True
        # elif pos[1] < -0.04:
        #     abort_flag = True
        elif abs(pos[2]) < 0.12:
            abort_flag = True
        else:
            abort_flag = False

        return abort_flag

    def check(self):
        pos, ori = self.robot_location()
        if abs(pos[0]) > 0.1:
            abort_flag = True
        elif (abs(ori[0]) > np.pi / 6) or (abs(ori[1]) > np.pi / 6) or (abs(ori[2]) > np.pi / 6):
            abort_flag = True
        # elif pos[1] < -0.04:
        #     abort_flag = True
        elif abs(pos[2]) < 0.13:
            abort_flag = True
        else:
            abort_flag = False

        return abort_flag



URDF_PTH = '../robot_zoo/no_idea/'
data_save_root = "/home/ubuntu/Documents/data_4_meta_self_modeling_id/sign_data/"

if __name__ == "__main__":

    # Train
    mode = 1
    Task = 17
    print(Task) # 13-17 right


    if Task<10:
        BEHAVIOR = 'f'
    elif Task>=10 and Task<13:
        BEHAVIOR = 'l'
    else:
        BEHAVIOR = 'r'

    friction_noise_Flag = True
    friction_noise_list = [0.9, 0.99,2.99]

    action_noise = 0.3
    if mode == 0:
        random_search_factor = 0.2
        Time_SLEEP_FLAG = False

    else:
        random_search_factor = 0
        Time_SLEEP_FLAG = True

    if mode == 0:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)

    para_config = np.loadtxt('../data/para_config.csv')

    initial_para = para_config[:, 0]
    para_range = para_config[:, 1:]
    all_robot_test = []
    done_times_log = []
    filtered_robot_list = []

    robot_name = "10_9_9_6_11_9_9_6_13_3_3_6_14_3_3_6"
    log_pth = data_save_root + "%s/" % robot_name

    if mode == 0:
        # loop_action = np.loadtxt(f'para_{BEHAVIOR}_{Task}.csv')
        loop_action = np.loadtxt('para_forward.csv')

        # loop_action = np.copy(initial_para)
    else:
        loop_action = np.loadtxt(f'para_{BEHAVIOR}_{Task}.csv')
        # loop_action = np.loadtxt('para_forward%d.csv'%Task)





    # initial_joints_angle = np.loadtxt(URDF_PTH + "%s/%s.txt" % (robot_name, robot_name))
    # initial_joints_angle = initial_joints_angle[0] if len(
    #     initial_joints_angle.shape) == 2 else initial_joints_angle
    # [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18] joint idx
    # initial_joints_angle = [0,0,0.8,0.2,0,
    #                         0,0,0.8,0.2,0,
    #                         0,0,0.8,0.2,0,
    #                         0,0,0.8,0.2,0,]
    initial_joints_angle = [0, 0, -0.8, -0.8, 0,
                            0, 0.5, -0.8, -0.8, 0,
                            0, 0.5, -0.8, -0.8, 0,
                            0, 0, -0.8, -0.8, 0, ]






    # [1, 2, 3, 4, 9, 11, 13, 14, 15, 16, 17, 22, 30, 31, 32, 34]


    meta_env = meta_sm_Env(initial_joints_angle,
                           urdf_path=URDF_PTH + "%s/%s.urdf" % (robot_name, robot_name))

    max_train_step = 100
    meta_env.sleep_time = 0
    obs = meta_env.reset()
    step_times = 0
    r_record = -np.inf
    done_time = 0
    done_time_killer = 3
    num_population = 20


    mu = 0.8

    while 1:
        # tunning 80%
        loop_action_array = np.repeat([loop_action], int(num_population * mu), axis=0)
        # keep one of previous best one
        loop_action_array[1:] = np.random.normal(loop_action_array[1:], scale=random_search_factor)
        # random_new 20%
        norm_space = np.random.sample((int(num_population - mu * num_population), len(initial_para)))
        action_list_append = norm_space * (para_range[:, 1] - para_range[:, 0]) + np.repeat(
            [para_range[:, 0]],
            int(num_population - mu * num_population), axis=0)
        # All individuals actions
        action_list = np.vstack((loop_action_array, action_list_append))

        rewards = []

        for i in range(num_population):
            obs = meta_env.resetBase()

            action = action_list[i]
            _, r, done, _ = meta_env.step(action)

            rewards.append(r)

            if r > r_record:
                r_record = r
                if mode == 0:
                    np.savetxt(f'para_{BEHAVIOR}_{Task}.csv', action)

        print("step count:", step_times, "r:", r_record)
        best_id = np.argmax(rewards)
        loop_action = action_list[best_id]
        np.savetxt('para_forward(debug).csv', loop_action)
        step_times += 1

    p.disconnect()
