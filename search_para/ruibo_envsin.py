import os
import time
import matplotlib.pyplot as plt
import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data as pd
import urdfpy as upy
from controller import *

'''
Description: Given the name of the robot, get the URDF file path and initial_joint_angle file path
Input: 
    robot_names: The unique names of the robot in list (like x_x_x_x_x_x_x_x_x_x_x_x_x_x_x_x)
    urdf_path: The file path stores urdf & initial joint angles files
Output:
    array([robot_name, urdf_path, initial_joints_angle])
'''


def getRobotURDFAndInitialJointAngleFilePath(robot_name, urdf_path):
    urdf_folder_list = os.listdir(path=urdf_path)

    if robot_name in urdf_folder_list:
        each_robot_urdf_file_name = robot_name + '.urdf'
        each_robot_joint_file_name = robot_name + '.txt'

        initial_urdf_path = os.path.join(
            urdf_path, robot_name, each_robot_urdf_file_name)
        initial_joints_path = os.path.join(
            urdf_path, robot_name, each_robot_joint_file_name)
        initial_joints_angle = np.loadtxt(
            initial_joints_path, max_rows=1).tolist()
        # print(f"The initial_joints_angle is {initial_joints_angle}")

        robot_name_urdf_joint = [robot_name,
                                 initial_urdf_path, initial_joints_angle]

        return robot_name_urdf_joint
    else:
        print('cannot find this robot!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        exit()


class RobotEnv(gym.Env):
    '''
    __init__, getObs(), _reset() are all functions define a robot and its current state
    '''

    def __init__(self, robot_name_urdf_joint, robot_start_pos, follow_robot=False):
        self.robot_name = robot_name_urdf_joint[0]
        self.urdf_path = robot_name_urdf_joint[1]
        self.initial_joints_angle = robot_name_urdf_joint[2]
        self.n_sim_steps = 30
        self.sleep_time = 1 / 480
        self.inner_motor_index = [1, 6, 11, 16]
        self.middle_motor_index = [2, 7, 12, 17]
        self.outer_motor_index = [3, 8, 13, 18]
        self.feet_index = [4, 9, 14, 19]
        self.num_leg = 4
        self.joints_index = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]
        self.initial_moving_joints_angle = np.asarray(
            [3 / np.pi * self.initial_joints_angle[idx] for idx in self.joints_index])

        self.camera_capture = follow_robot
        self.mode = p.POSITION_CONTROL
        self.max_velocity = 1.8
        self.force = 1.6
        # self.n_sim_steps = 30
        self.motor_action_space = np.pi / 3
        self.friction = 0.99
        self.ik_solver = 0
        self.robot_start_pos = robot_start_pos
        self.robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.delta_feet_xyz = np.asarray(
            [[0.0, 0.05, 0.0] for i in range(self.num_leg)])
        # for ppo
        self.num_steps = 0
        self.observation_space = spaces.Box(
            low=-np.pi / 3, high=np.pi / 3, shape=(18,))
        self.action_space = spaces.Box(
            low=-2 * np.pi, high=2 * np.pi, shape=(10,))
        # getDynamicsInfo

        # for ppo
        # self.reset(friction)

    def reset(self, friction = 0.99):
        p.setTimeStep(1/240)
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, -9.81)
        planeId = p.loadURDF("plane.urdf", [0, 0, 0])
        self.robotId = p.loadURDF(self.urdf_path, self.robot_start_pos, self.robot_start_orientation,
                                  flags=p.URDF_USE_SELF_COLLISION, useFixedBase=0)
        self.dof = p.getNumJoints(self.robotId)
        # self.getBaseDynamics = p.getDynamicsInfo(self.robotId, 10)
        # print("aaaa", self.getBaseDynamics)
        change_mass = []
        # p.changeDynamics(self.robotId, 9, mass=0.1)
        # p.changeDynamics(self.robotId, 14, mass=0.1)
        # # p.changeDynamics(self.robotId, 19, mass=0.1)
        # # p.changeDynamics(self.robotId, 4, mass=0.1)
        #
        # p.changeDynamics(self.robotId, 5, mass=0.15)
        # p.changeDynamics(self.robotId, 10, mass=0.15)

        self.friction = friction
        # self.force = force
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)

        for i in range(len(self.initial_joints_angle)):
            p.resetJointState(self.robotId, i, self.initial_joints_angle[i])

        for _ in range(1000):
            p.stepSimulation()
            # time.sleep(1/240)

        self.cur_base_pos, self.cur_base_ori = p.getBasePositionAndOrientation(
            self.robotId)
        self.cur_base_ori = p.getEulerFromQuaternion(self.cur_base_ori)
        self.cur_base_pos, self.cur_base_ori = np.array(
            self.cur_base_pos), np.array(self.cur_base_ori)

        # ppo
        self.num_steps = 0

        return self.get_obs()

    def get_obs(self):
        self.last_base_pos = self.cur_base_pos
        self.last_base_ori = self.cur_base_ori

        self.cur_base_pos, self.cur_base_ori = p.getBasePositionAndOrientation(
            self.robotId)
        self.cur_base_ori = p.getEulerFromQuaternion(self.cur_base_ori)
        self.cur_base_pos, self.cur_base_ori = np.array(
            self.cur_base_pos), np.array(self.cur_base_ori)

        self.delta_base_pos = self.cur_base_pos - self.last_base_pos
        self.delta_base_ori = self.cur_base_ori - self.last_base_ori

        self.cur_joints_angle = np.asarray(p.getJointStates(self.robotId, self.joints_index))[:,
                                                                                              0]  # only get joint position

        obs = np.concatenate(
            [self.delta_base_pos, self.delta_base_ori, self.cur_joints_angle])
        return obs

    def robot_location(self):
        #     if call this, the absolute location of robot body will back
        position, orientation = p.getBasePositionAndOrientation(self.robotId)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()
        # if abs(pos[0]) > 0.5:
        #     print("Fail: pos_x", pos[0])
        #     abort_flag = True
        # elif abs(ori[1]) > np.pi / 3 or abs(ori[2]) > np.pi / 3:
        #     abort_flag = True
        #     print("Fail: angle", ori[1], ori[2])
        # elif abs(pos[2]) < 0.08:

        if abs(pos[2]) < 0.08:
            # print("Fail: pos_z", pos[2])
            abort_flag = True
        elif abs(ori[0] > np.pi / 3) or abs(ori[1] > np.pi / 3) or abs(ori[2] > 3):
            # print("Fail: ori", ori[:2])
            abort_flag = True
        else:
            abort_flag = False
        return abort_flag

    def act(self, joints_action):
        for i in range(len(joints_action)):
            p.setJointMotorControl2(self.robotId, self.joints_index[i], controlMode=self.mode,
                                    targetPosition=joints_action[i], force=self.force, maxVelocity=self.max_velocity)
        for i in range(self.n_sim_steps):
            p.stepSimulation()  # 30 times, SLEEP_T * 30 = consumed time
            if GUI_flag == True:
                if self.camera_capture == True:
                    basePos, baseOrn = p.getBasePositionAndOrientation(
                        self.robotId)  # Get model position
                    basePos_list = [basePos[0], basePos[1], 0.3]
                    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                                                 cameraTargetPosition=basePos_list)  # fix camera onto model

                # time.sleep(self.sleep_time)

    def step(self, sin_para):
        # frict = []
        # print(sin_para)
        # command = []
        # real_move = []
        self.num_steps += 1
        done_flag = False
        p_energy = self.initial_moving_joints_angle
        sum_energy = 0
        # self.friction = random.uniform(0.5, 1)
        # frict = self.friction
        for i in range(Period):
            # self.friction = random.uniform(0.5, 1)
            # frict.append(self.friction)
            action_add = sin_move(i, sin_para)
            action = self.initial_moving_joints_angle + action_add
            action = np.clip(action, -1, 1)
            action *= self.motor_action_space
            sum_energy += sum((action - p_energy)**2)
            p_energy = action
            self.act(action)
            step_pos, step_ori = self.robot_location()
            done = self.check()
            if done:
                done_flag = True

        obs = self.get_obs()

        # r = 100 * obs[1] - 50 * np.abs(obs[0])
        r = 0
        # return obs, r, done_flag, {}
        return obs, r, done_flag, None, sum_energy

    def step2(self, sin_para):
        command = []
        real_move = []
        self.num_steps += 1
        done_flag = False
        for i in range(Period):
            action_add = sin_move(i, sin_para)
            action = self.initial_moving_joints_angle + action_add
            action = np.clip(action, -1, 1)
            action *= self.motor_action_space
            self.act(action)
            joints_angle = self.get_obs()[6:]
            real_move.append(joints_angle)
            command.append(action)
            step_pos, step_ori = self.robot_location()
            done = self.check()
            if done:
                done_flag = True

        obs = self.get_obs()

        # r = 100 * obs[1] - 50 * np.abs(obs[0])
        r = 0
        return obs, r, done_flag, {}, np.asarray(command), np.asarray(real_move)
        # return obs, r, done_flag, {}, frict

    def step_check(self, sin_para, friction):
        # print(sin_para)
        self.num_steps += 1
        done_flag = False
        self.friction = friction
        for i in range(Period):
            # self.friction = friction[i]
            # print(self.friction)
            action_add = sin_move(i, sin_para)
            action = self.initial_moving_joints_angle + action_add
            action = np.clip(action, -1, 1)
            action *= self.motor_action_space
            self.act(action)
            step_pos, step_ori = self.robot_location()
            done = self.check()
            if done:
                done_flag = True

        obs = self.get_obs()

        # r = 100 * obs[1] - 50 * np.abs(obs[0])
        r = 0

        return obs, r, done_flag, {}


def standRobot(stand_robot_path, idx):
    stand_robots_name = sorted(np.loadtxt(
        stand_robot_path, dtype=np.str, max_rows=1061, delimiter='\n'))
    print(len(stand_robots_name))
    robot_name = stand_robots_name[idx]
    return robot_name



# STAND_ROBOT_PATH = '../data/robot_name_list.txt'
URDF_JOINT_FILE_PATH = '../data/robot_urdf_search/'
GUI_flag = True

if __name__ == '__main__':
    if GUI_flag == True:
        physicsClient = p.connect(1)  # p.GUI = 1
    else:
        physicsClient = p.connect(2)  # p.DIRECT = 2

    ROBOTID = 1
    epoch_num = 1

    # robot_name = standRobot(STAND_ROBOT_PATH, ROBOTID)

    # robot_name = '6_0_0_3_11_4_11_10_13_8_1_2_9_0_0_9'
    # robot_name = '11_1_11_4_15_10_2_0_19_2_10_0_13_11_1_8'

    # print(f"The robot name is {robot_name}\n")

    robot_names = os.listdir("../data/robot_urdf_search")
    for robot_name in robot_names:
        robot_info = getRobotURDFAndInitialJointAngleFilePath(
            robot_name, urdf_path=URDF_JOINT_FILE_PATH)

        # para = np.loadtxt(os.path.join(URDF_JOINT_FILE_PATH, robot_name) + "/para_mode0_diffenv_test.csv")

        # robot name, urdf_path, initial_joints_angle
        env = RobotEnv(robot_info, [0, 0, 0.3], follow_robot=False)
        env.sleep_time = 1 / 480

        fail = 0
        result = 0
        action_logger = []
        joint_state_logger = []

        env.reset()
        for step_i in range(10):
            sin_para = random_para()
            # sin_para = para
            action_logger.append(sin_para)

            obs, r, done,_  = env.step(sin_para)
            print(obs)
            joint_state_logger.append(obs[6:])

            # print(env.robot_location())
            result += r
        action_logger = np.asarray(action_logger)
        joint_state_logger = np.asarray(joint_state_logger)
        # for i in range(12):
        #     plt.plot(range(160), joint_state_logger[:,i])
        #     plt.plot(range(160),action_logger[:,i])
        #     plt.show()
