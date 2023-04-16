import os
import time
import random
import pickle

import gym
import torch
import numpy as np
import pybullet as p
from tqdm import tqdm
import pybullet_data as pd

from search_para.controller import sin_move, random_para
from dynamic_representation import SELayer, PredConf, MLSTMfcn, unique_leg_conf_idx
from create_urdf.cal_single import *


class meta_sm_Env(gym.Env):
    def __init__(self, init_q, render, noise_para, data_save_pth, robot_camera=False, urdf_path='CAD2URDF'):
        self.save_pth = data_save_pth

        self.robotid = None
        self.q = None
        self.p = None
        self.last_obs = [0] * 18
        self.obs = [0] * 18
        self.mode = p.POSITION_CONTROL
        self.sleep_time = 1. / 240  # decrease the value if it is too slow.

        self.maxVelocity = 1.5  # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.force = 1.8

        self.n_sim_steps = 30
        self.motor_action_space = np.pi / 3
        self.urdf_path = urdf_path
        self.camera_capture = False
        self.robot_camera = robot_camera
        self.friction = 0.99
        self.robot_view_path = None
        self.sub_step_num = 16
        self.joint_moving_idx = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]
        self.initial_moving_joints_angle = np.asarray(
            [3 / np.pi * init_q[idx] for idx in self.joint_moving_idx])

        self.robot_type = [[3, 6],
                           [0, 3, 6, 9],
                           [0, 3, 4, 6, 7, 9, ],
                           [0, 1, 3, 4, 6, 7, 9, 10],
                           [0, 1, 3, 4, 5, 6, 7, 8, 9, 10],
                           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

        self.noise_para = noise_para
        self.action_space = gym.spaces.Box(low=-np.ones(16, dtype=np.float32), high=np.ones(16, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=-np.ones(18, dtype=np.float32) * np.inf,
                                                high=np.ones(18, dtype=np.float32) * np.inf)

        self.state_dynamics = []
        self.action_dynamics = []
        self.count = 0

        p.setAdditionalSearchPath(pd.getDataPath())

    def get_obs(self):
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        jointInfo = [p.getJointState(self.robotid, i) for i in self.joint_moving_idx]
        jointVals = np.array([[joint[0]] for joint in jointInfo]).flatten()
        self.obs = np.concatenate([self.p, self.q, jointVals])
        return self.obs

    def act(self, sin_para):
        self.action_dynamics.append(sin_para)
        # sin_para = sin_para*self.noise_para + self.para

        for sub_step in range(self.sub_step_num):
            a_add = sin_move(sub_step, sin_para)
            a = self.initial_moving_joints_angle + a_add
            a = np.clip(a, -1, 1)
            a *= self.motor_action_space

            for i in range(12):
                pos_value = a[i]
                p.setJointMotorControl2(self.robotid, self.joint_moving_idx[i], controlMode=self.mode,
                                        targetPosition=pos_value,
                                        force=self.force,
                                        maxVelocity=self.maxVelocity)
            for i in range(self.n_sim_steps):
                p.stepSimulation()
                if self.render:
                    # Capture Camera
                    if self.camera_capture == True:
                        basePos, baseOrn = p.getBasePositionAndOrientation(self.robotid)  # Get model position
                        basePos_list = [basePos[0], basePos[1], 0.3]
                        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=75, cameraPitch=-20,
                                                     cameraTargetPosition=basePos_list)  # fix camera onto model
                time.sleep(self.sleep_time)
            self.state_dynamics.append(self.get_obs())

    def reset(self):
        self.state_dynamics = []
        self.action_dynamics = []
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)

        robotStartPos = [0, 0, 0.3]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robotid = p.loadURDF(self.urdf_path, robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=0)

        for i in range(12):
            pos_value = self.initial_moving_joints_angle[i]
            p.setJointMotorControl2(self.robotid, self.joint_moving_idx[i], controlMode=self.mode,
                                    targetPosition=pos_value, force=self.force, maxVelocity=100)

        for _ in range(60):
            p.stepSimulation()

        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        init_state = self.get_obs()
        self.state_dynamics.append(init_state)
        return init_state

    def step(self, a):
        self.act(a)
        obs = self.get_obs()

        pos, _ = self.robot_location()
        r = pos[1]
        done = self.check()
        self.count += 1
        return obs, r, done, {}

    def robot_location(self):
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()
        # if abs(pos[0]) > 0.5:
        #     abort_flag = True
        if abs(ori[0]) > np.pi / 2 or abs(ori[1]) > np.pi / 2:
            abort_flag = True
        # elif abs(pos[2]) < 0.22:
        #     abort_flag = True
        else:
            abort_flag = False
        return abort_flag

def replay_robot_awareness(robot_name, robot_path, action_data, pred_urdf_code_list):
    data_path = robot_path[:robot_path.rindex('/')+1]
    initial_joints_angle = np.loadtxt(data_path + "%s.txt" % robot_name)
    initial_joints_angle = initial_joints_angle[0] if len(
        initial_joints_angle.shape) == 2 else initial_joints_angle
    para_space = np.asarray([0.6, 0.6,
                             0.6, 0.6, 0.6, 0.6,
                             0.5, 0.5,
                             0.6, 0.6,
                             0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    # para = np.loadtxt(data_path + "para_mode0_diffenv_test.csv")

    render_flag = True
    env = meta_sm_Env(initial_joints_angle, render_flag, para_space,
                      urdf_path=data_path + "%s.urdf" % robot_name, data_save_pth=data_path)
    env.sleep_time = 1/340
    # env.sleep_time = 0
    obs = env.reset()
    prev_id = None
    text_id = None
    p.addUserDebugText(f'Robot Shape Code: {robot_name}', [-0.2, 0, 0.6], textSize=2, textColorRGB=[0, 0, 150])
    # p.addUserDebugText(f'Robot Shape Code: {robot_name}', [-0.2, 0, 0.6], textSize=2, textColorRGB=[244, 187, 0])
    time.sleep(10)
    for i in range(len(pred_urdf_code_list)):
        action = action_data[i]
        next_obs, r, done, _ = env.step(action)

        pred_urdf_code = pred_urdf_code_list[i]
        pred_robot_name = '_'.join(map(str, pred_urdf_code))
        position, orientation = p.getBasePositionAndOrientation(env.robotid)

        error = np.linalg.norm(np.array(pred_urdf_code) - np.array(list(map(int, robot_name.split('_')))), ord=1)

        # write_urdf(pred_urdf_code[0:4], pred_urdf_code[4:8], pred_urdf_code[8:12], pred_urdf_code[12:], 'temp_urdf/')

        if prev_id is not None:
            p.removeBody(prev_id)
            p.removeUserDebugItem(text_id)
            p.removeUserDebugItem(text_id_2)

        tmp_id = p.loadURDF(os.path.join('temp_urdf', pred_robot_name, f"{pred_robot_name}_nocol.urdf"), position, orientation)
        for j in env.joint_moving_idx:
            joint_state = p.getJointState(env.robotid, j)[0]
            p.resetJointState(tmp_id, j, joint_state)

        text_id = p.addUserDebugText(f'Step {i+1}, Pred Shape Code: {pred_robot_name}', [-0.2, 0, 0.5], textSize=2, textColorRGB=[0, 100, 0])

        text_id_2 = p.addUserDebugText(f'L1 distance {int(error)}', [-0.2, 0, 0.4], textSize=2,
                                     textColorRGB=[150, 0, 0])
        time.sleep(2)

        prev_id = tmp_id

    input()





if __name__ == '__main__':
    render = True
    if render:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    robot_name, robot_path, action_data, pred_urdf_code_list = pickle.load(open('aware_robots/11_9_9_2_10_10_7_6_14_2_5_6_13_3_3_10.pkl', 'rb'))
    replay_robot_awareness(robot_name, robot_path, action_data, pred_urdf_code_list)


