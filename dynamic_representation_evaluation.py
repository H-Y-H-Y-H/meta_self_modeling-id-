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

def collect_robot_dynamics(robot_name, robot_path, total_step=20):
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

    render_flag = False
    env = meta_sm_Env(initial_joints_angle, render_flag, para_space,
                      urdf_path=data_path + "%s.urdf" % robot_name, data_save_pth=data_path)
    # env.sleep_time = 1/1000
    env.sleep_time = 0

    obs = env.reset()
    fall = False
    for i in range(total_step):
        action = random_para()
        next_obs, r, done, _ = env.step(action)
        if done:
            fall = True
            break
        obs = next_obs
    dynamics = np.vstack(env.state_dynamics).astype(dtype=np.float32)[1:][None, :]
    actions = np.vstack(env.action_dynamics).astype(dtype=np.float32)
    if fall:
        dynamics = dynamics[:, :-16, :]
        actions = actions[:-1, :]
    return dynamics, actions


def find_an_aware_robot(robot_paths, model, idx2leg):
    for robot_name, robot_path in tqdm(robot_paths.items()):
        dynamic_data, action_data = collect_robot_dynamics(robot_name, robot_path)
        steps = dynamic_data.shape[1] // 16
        step_pred_diff = []
        pred_urdf_code_list = []
        for s in range(1, steps+1):
            name_code = list(map(int, robot_name.split('_')))
            leg_conf = (name_code[0], name_code[4], name_code[8], name_code[12])
            joint_conf = np.array(name_code[1:4] + name_code[5:8])

            length = torch.tensor([s * 16])
            dynamic_torch = torch.from_numpy(dynamic_data[:, :length, :])

            with torch.no_grad():
                leg_logit, pred_joint_confs_logit = model(dynamic_torch, length)
                pred_leg_label = torch.argmax(leg_logit, dim=1).detach().cpu().item()
                pred_joint_conf = np.hstack([torch.argmax(pred_joint_confs_logit[i], dim=1).detach().cpu().numpy() for i in range(6)])
            pred_leg = idx2leg[pred_leg_label]
            diff = np.linalg.norm(pred_joint_conf - joint_conf, ord=1)
            step_pred_diff.append(diff)

            mirror_joint_conf = (12 - pred_joint_conf) % 12
            pred_urdf_code = [pred_leg[0]] + list(pred_joint_conf[:3]) + \
                             [pred_leg[1]] + list(pred_joint_conf[3:]) + \
                             [pred_leg[2]] + list(mirror_joint_conf[3:]) + \
                             [pred_leg[3]] + list(mirror_joint_conf[:3])
            pred_urdf_code_list.append(pred_urdf_code)

            if diff == 0 and pred_leg == leg_conf:
                assert robot_name == '_'.join(map(str, pred_urdf_code))
                data = (robot_name, robot_path, action_data, pred_urdf_code_list)
                pickle.dump(data, open(f"aware_robots/{robot_name}.pkl", 'wb'))
                import sys
                sys.exit()

        # print(pred_leg == leg_conf, [float(f"{spd:.4f}") for spd in step_pred_diff])

def evaluate_model(robot_list, model, idx2leg):
    """
    eval_result: (bool: has_aware, int: step_of_aware, list: l1_diff, list of np.array: latents, list: predict_codes)
    """
    for robot_name in tqdm(robot_list):
        # dynamic_data_path = os.path.join('data', 'robot_urdf_80k', robot_name, "random_dynamic_step20_untilfail.npy")
        dynamic_data_path = os.path.join('data', 'robot_urdf_80k', robot_name, "random_dynamic_step20_example8_untilfail.pkl")
        state_list = pickle.load(open(dynamic_data_path, 'rb'))

        state_list_short = []
        for state in state_list:
            state = state[1:]
            state_list_short.append(state)

        state_list_correct = [state_list_short[0]]
        prev_len = state_list_short[0].shape[0]
        for state in state_list_short[1:]:
            state_correct = state[prev_len:]
            prev_len = state.shape[0]
            state_list_correct.append(state_correct)

        longest_dynamic = max(state_list_correct, key=lambda x:x.shape[0])

        dynamic_data = longest_dynamic[None, 1:]
        # dynamic_data = np.load(dynamic_data_path)[None, :-1]

        steps = dynamic_data.shape[1] // 16
        step_pred_diff = []
        latents = []
        predict_codes = []
        for s in range(1, steps+1):
            name_code = list(map(int, robot_name.split('_')))
            leg_conf = (name_code[0], name_code[4], name_code[8], name_code[12])
            joint_conf = np.array(name_code[1:4] + name_code[5:8])

            length = torch.tensor([s * 16])
            dynamic_torch = torch.from_numpy(dynamic_data[:, :length, :]).cuda()

            with torch.no_grad():
                leg_logit, pred_joint_confs_logit = model(dynamic_torch, length)
                latent = model.get_latent(dynamic_torch, length)[0].detach().cpu().numpy()
                pred_leg_label = torch.argmax(leg_logit, dim=1).detach().cpu().item()
                pred_joint_conf = np.hstack(
                    [torch.argmax(pred_joint_confs_logit[i], dim=1).detach().cpu().numpy() for i in range(6)])
            pred_leg = idx2leg[pred_leg_label]
            diff = np.linalg.norm(pred_joint_conf - joint_conf, ord=1)
            step_pred_diff.append(diff)
            latents.append(latent)

            mirror_joint_conf = (12 - pred_joint_conf) % 12
            pred_urdf_code = [pred_leg[0]] + list(pred_joint_conf[:3]) + \
                             [pred_leg[1]] + list(pred_joint_conf[3:]) + \
                             [pred_leg[2]] + list(mirror_joint_conf[3:]) + \
                             [pred_leg[3]] + list(mirror_joint_conf[:3])
            predict_codes.append(pred_urdf_code)

        has_aware = 0 in step_pred_diff
        step_of_aware = None
        if has_aware:
            step_of_aware = step_pred_diff.index(0) + 1
        eval_result = (has_aware, step_of_aware, step_pred_diff, latents, predict_codes)
        eval_file_path = os.path.join('data', 'robot_urdf_80k', robot_name, "eval_result_2.pkl")
        pickle.dump(eval_result, open(eval_file_path, 'wb'))


if __name__ == '__main__':
    # robot_names = []
    # robot_paths = dict()
    # robot_name_path = open('data/robot_name_list_260k.txt').read().strip().split('\n')
    # for rnp in robot_name_path:
    #     rn, rp = rnp.split()
    #     robot_names.append(rn)
    #     robot_paths[rn] = rp

    robot_names = []
    for r in os.listdir('data/robot_urdf_80k'):
        rb_files = os.listdir(os.path.join('data', 'robot_urdf_80k', r))
        if 'random_dynamic_step20_example8_untilfail.pkl' in rb_files and "eval_result_2.pkl" not in rb_files:
            robot_names.append(r)
    robot_names.sort()
    print("Robot count:", len(robot_names))

    unique_leg_count = unique_leg_conf_idx(robot_names)
    idx2leg = list(unique_leg_count.keys())
    leg2idx = {leg: idx for idx, leg in enumerate(idx2leg)}

    # render = False
    # if render:
    #     p.connect(p.GUI)
    # else:
    #     p.connect(p.DIRECT)

    model = PredConf(do=0, mlp_hidden_dim=256)
    model.load_state_dict(torch.load("log_dynamic_varying_sample_size/epoch693-acc0.5613"))
    model.eval()
    model = model.cuda()
    # find_an_aware_robot(robot_paths, model, idx2leg)

    evaluate_model(robot_names, model, idx2leg)




