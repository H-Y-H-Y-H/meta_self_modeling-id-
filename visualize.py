import os
import math as m
import time
import pybullet_data as pd
import gym
from search_para.controller import *
import numpy as np


np.random.seed(2022)

class robot_zoo(gym.Env):
    def __init__(self, robot_list_name,robot_camera=False):

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

        self.n_sim_steps = 30
        self.motor_action_space = np.pi / 3
        self.camera_capture = False
        self.robot_camera = robot_camera
        self.friction = 0.99
        self.robot_view_path = None
        self.sub_step_num = 16
        self.joint_moving_idx = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]

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

        # p.changeVisualShape(planeId, -1, rgbaColor=[0.2, 0.2, 0.2, 1])

        robotStartPos = [0, 0, 0.3]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        length = 10
        self.robotid = []
        random.seed(0)

        random.shuffle(robot_name_list)

        for i in range(1):

            robot_name = robot_name_list[i]

            init_q = np.loadtxt(URDF_PTH + "%s/%s.txt" % (robot_name, robot_name))

            robotStartPos = [(i%length), (i//length),0.3]
            urdfpath = URDF_PTH + "%s/%s.urdf" % (robot_name, robot_name)
            self.robotid= p.loadURDF(urdfpath ,robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=0)
            # text_id = p.addUserDebugText("Robot: "+ robot_name, [(i%length), (i//width)-0.5,0.5],
            #                              lifeTime=0,
            #                              textSize=1,
            #                              textColorRGB=[0, 0, 0])
            self.initial_moving_joints_angle = np.asarray([3 / np.pi * init_q[idx] for idx in self.joint_moving_idx])

            p.changeDynamics(self.robotid, 2, lateralFriction=self.friction)
            p.changeDynamics(self.robotid, 5, lateralFriction=self.friction)
            p.changeDynamics(self.robotid, 8, lateralFriction=self.friction)
            p.changeDynamics(self.robotid, 11, lateralFriction=self.friction)

            p.changeVisualShape(self.robotid, -1, rgbaColor=[118/255, 182/255, 238/255, 1])

            for link in [0,2,
                         5,7,
                         10,12,
                         15,17]:
                p.changeVisualShape(self.robotid, link, rgbaColor=[2/255, 33/255, 105/255, 1])

            for link in [1,3,6,8,11,13,16,18]:
                p.changeVisualShape(self.robotid, link, rgbaColor=[118/255, 182/255, 238/255, 1])


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

    def step(self, sin_para):

        step_action_list = []
        step_obs_list = []
        r = 0
        # range 1 to step_num +1 so that the robot can achieve the original pos after current action.
        # for a_i in range(1, self.sub_step_num + 1):
        #     a_i_add = sin_move(a_i, sin_para)
        #     norm_a = self.initial_moving_joints_angle + a_i_add
        #     norm_a = np.clip(norm_a, -1, 1)
        #     a = norm_a * self.motor_action_space
        #     self.act(a)
        #     step_obs = self.get_obs()
        #     step_obs_list.append(step_obs)
        #     step_action_list.append(norm_a)  # from -1 to 1
        #     r += (3 * step_obs[1] - abs(step_obs[5]) - 0.5 * abs(step_obs[0]) + 1)
        # step_done = self.check()
        step_done = 0
        self.count += 1
        obs_combine = np.hstack((step_action_list, step_obs_list))
        return obs_combine, r, step_done, {}

    def robot_location(self):
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()
        # if abs(pos[0]) > 0.5:
        #     abort_flag = True
        if (abs(ori[0]) > np.pi / 6) or (abs(ori[1]) > np.pi / 6):  # or abs(ori[2]) > np.pi / 6:
            abort_flag = True
        # elif pos[1] < -0.04:
        #     abort_flag = True
        # elif abs(pos[0]) > 0.2:
        #     abort_flag = True
        else:
            abort_flag = False
        return abort_flag

p.connect(p.GUI)

URDF_PTH = 'robot_zoo/'
# robot_name_list = np.loadtxt('meta_sm/test_results/100acc_robo_name.txt',dtype='str')
robot_name_list = os.listdir(URDF_PTH)
# robot_name_list = ["11_0_2_0_10_0_9_2_14_0_3_10_13_0_10_0"]*2

para_config = np.loadtxt('data/para_config.csv')
initial_para = para_config[:, 0]
para_range = para_config[:, 1:]



meta_env = robot_zoo(robot_name_list)
meta_env.reset()
while 1:
    a = 0
    p.stepSimulation()
    time.sleep(1/240)


