import os

from torch.utils.data import Dataset, DataLoader
import time
import pybullet_data as pd
import gym
from search_para.controller import *
import numpy as np
from model import *
from multiprocessing import Process
from tqdm import tqdm

random.seed(2022)
np.random.seed(2022)


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

        self.n_sim_steps = 30
        self.motor_action_space = np.pi / 3
        self.urdf_path = urdf_path
        self.camera_capture = False
        self.robot_camera = robot_camera
        self.friction = 0.99
        self.robot_view_path = None
        self.sub_step_num = 16
        self.joint_moving_idx = [1, 2, 3, 6, 7, 8, 11, 12, 13, 16, 17, 18]
        self.initial_moving_joints_angle = np.asarray([3 / np.pi * init_q[idx] for idx in self.joint_moving_idx])

        self.action_space = gym.spaces.Box(low=-np.ones(16, dtype=np.float32), high=np.ones(16, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=-np.ones(18, dtype=np.float32) * np.inf,
                                                high=np.ones(18, dtype=np.float32) * np.inf)

        self.log_obs = []
        self.log_action = []
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

    def act(self, sin_para):
        for sub_step in range(self.sub_step_num):
            a_add = sin_move(sub_step, sin_para)
            a = self.initial_moving_joints_angle + a_add
            a = np.clip(a, -1, 1)
            a *= self.motor_action_space

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
            time.sleep(self.sleep_time)

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

        p.changeDynamics(self.robotid, 2, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 5, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 8, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 11, lateralFriction=self.friction)

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

    def step(self, a):

        self.act(a)
        obs = self.get_obs()

        r = 3 * obs[1] - abs(obs[5]) - 0.5 * abs(obs[0]) + 1
        Done = self.check()

        self.count += 1

        return obs, r, Done, {}

    def robot_location(self):
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()
        # if abs(pos[0]) > 0.5:
        #     abort_flag = True
        if abs(ori[0]) > np.pi / 6 or abs(ori[1]) > np.pi / 6: # or abs(ori[2]) > np.pi / 6:
            abort_flag = True
        # elif pos[1] < -0.04:
        #     abort_flag = True
        # elif abs(pos[0]) > 0.2:
        #     abort_flag = True
        else:
            abort_flag = False
        return abort_flag


def collect_dyna_sm_data(env, sm_model, choose_a, use_policy, train_data=None, test_data=None, num_candidate=1000,
                         step_num=6):
    S, A, NS, R, D, select_ra_ornot = [], [], [], [], [], []
    obs = env.reset()

    if use_policy == 0:
        # Random action:
        for step in range(step_num):
            a = np.random.uniform(-1, 1, size=10)

            obs_, r, Done, _ = env.step(a)
            A.append(a)
            S.append(obs)
            NS.append(obs_)
            D.append(Done)
            R.append(r)
            obs = obs_

            if Done:
                obs = env.reset()
                # print(Done, step)

    elif use_policy == 1:
        sm_model.eval()
        pred_loss = []

        # Gaussian action:torch
        for step in range(step_num):
            A_array0 = np.asarray([choose_a] * (num_candidate // 2))
            A_array1 = np.random.uniform(-1, 1, size=((num_candidate // 2), 10))
            A_array = np.vstack((choose_a, A_array0, A_array1))
            A_array_numpy = np.random.normal(A_array, 0.2)
            S_array = np.asarray([obs] * (num_candidate + 1))
            S_array = torch.from_numpy(S_array.astype(np.float32)).to(device)
            A_array = torch.from_numpy(A_array_numpy.astype(np.float32)).to(device)
            pred_ns = sm_model.forward(S_array, A_array)
            pred_ns_numpy = pred_ns[0].cpu().detach().numpy()

            # Task:
            all_a_rewards = 3 * pred_ns_numpy[:, 1] - abs(pred_ns_numpy[:, 5]) - 0.5 * abs(pred_ns_numpy[:, 0]) \
                            - 0.5 * 2 ** (abs(pred_ns_numpy[:, 4]))  # avoid flipping.

            greedy_select = int(np.argmax(all_a_rewards))
            if greedy_select > 500:
                select_ra_ornot.append(0)
            elif greedy_select == 0:
                select_ra_ornot.append(2)
            else:
                select_ra_ornot.append(1)

            choose_a = A_array_numpy[greedy_select]
            pred = pred_ns_numpy[greedy_select]

            obs_, r, Done, _ = env.step(choose_a)

            gt = obs_
            loss = np.mean((gt - pred) ** 2)
            pred_loss.append(loss)

            A.append(choose_a)
            S.append(obs)
            NS.append(obs_)
            D.append(Done)
            obs = np.copy(obs_)
            R.append(r)

            if Done:
                obs = env.reset()
                choose_a = call_max_reward_action(train_data, test_data)
                choose_a = choose_a.cpu().detach().numpy()

    S, A, NS, R, D, select_ra_ornot = np.array(S), np.array(A), np.array(NS), np.array(R), np.array(D), np.array(
        select_ra_ornot)

    train_data_num = int(step_num * 0.8)
    idx_list = np.arange(step_num)
    np.random.shuffle(idx_list)
    train_SAS = [S[idx_list][:train_data_num],
                 A[idx_list][:train_data_num],
                 NS[idx_list][:train_data_num],
                 D[idx_list][:train_data_num],
                 R[idx_list][:train_data_num]]
    test_SAS = [S[idx_list][train_data_num:],
                A[idx_list][train_data_num:],
                NS[idx_list][train_data_num:],
                D[idx_list][train_data_num:],
                R[idx_list][train_data_num:]]

    return train_SAS, test_SAS, choose_a, select_ra_ornot


def call_max_reward_action(train_data, test_data):
    action_rewards = torch.stack((train_data.R, test_data.R))
    a_id = torch.argmax(action_rewards)
    a_choose = torch.stack((train_data.A, test_data.A))[a_id]
    return a_choose


class SAS_data(Dataset):
    def __init__(self, SAS_data):
        self.all_S = SAS_data[0]
        self.all_A = SAS_data[1]
        self.all_NS = SAS_data[2]
        self.all_DONE = SAS_data[3]
        self.all_R = SAS_data[4]

    def __getitem__(self, idx):
        S = self.all_S[idx]
        A = self.all_A[idx]
        NS = self.all_NS[idx]
        R = np.asarray([self.all_R[idx]])
        # DONE = np.asarray([self.all_DONE[idx]])

        self.S = torch.from_numpy(S.astype(np.float32)).to(device)
        self.A = torch.from_numpy(A.astype(np.float32)).to(device)
        self.NS = torch.from_numpy(NS.astype(np.float32)).to(device)
        self.R = torch.from_numpy(R.astype(np.float32)).to(device)
        # self.DONE = torch.from_numpy(DONE.astype(np.float32)).to(device)

        sample = {'S': self.S, 'A': self.A, "NS": self.NS, "R": self.R}

        return sample

    def __len__(self):
        return len(self.all_S)

    def add_data(self, SAS_data_):
        S_data, A_data, NS_data, all_DONE, all_R = SAS_data_
        self.all_S = np.vstack((self.all_S, S_data))
        self.all_A = np.vstack((self.all_A, A_data))
        self.all_NS = np.vstack((self.all_NS, NS_data))
        self.all_DONE = np.hstack((self.all_DONE, all_DONE))
        self.all_R = np.hstack((self.all_R, all_R))


def train_dyna_sm(sm_model, train_dataset, test_dataset):
    min_loss = + np.inf
    abort_learning = 0
    decay_lr = 0
    num_epoch = 100
    batchsize = 16
    all_train_L, all_valid_L = [], []

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(sm_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=False)

    for epoch in range(num_epoch):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        sm_model.train()
        for batch in train_dataloader:
            S, A, NS = batch["S"], batch["A"], batch["NS"]
            pred_NS = sm_model.forward(S, A)
            loss = sm_model.loss(pred_NS, NS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        sm_model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                S, A, NS = batch["S"], batch["A"], batch["NS"]
                pred_NS = sm_model.forward(S, A)
                loss = sm_model.loss(pred_NS, NS)
                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)

        if avg_valid_L < min_loss:
            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L
            # PATH = log_path + '/best_model.pt'
            # torch.save(sm_model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1
            decay_lr += 1
        # scheduler.step(avg_valid_L)
        # np.savetxt(log_path + "training_L.csv", np.asarray(all_train_L))
        # np.savetxt(log_path + "testing_L.csv", np.asarray(all_valid_L))

        if abort_learning > 5:
            break
        t1 = time.time()
        # print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", lr)

    # print("valid_loss:", min_loss)
    return min_loss, sm_model


def  collect_robot_sasf(robot_name, URDF_PTH, data_save_PTH, num_epochs = 50,
    NUM_EACH_CYCLE = 6, render_flag=False):


    robot_info_pth = URDF_PTH + "%s/" % robot_name
    initial_joints_angle = np.loadtxt(robot_info_pth + "%s.txt" % robot_name)
    initial_joints_angle = initial_joints_angle[0] if len(
        initial_joints_angle.shape) == 2 else initial_joints_angle

    env = meta_sm_Env(initial_joints_angle,
                      urdf_path=robot_info_pth + "%s.urdf" % robot_name)
    env.sleep_time = 0
    sm_model = FastNN(18 + 10, 18)
    # pretrain sm:
    # if load_pretrain_sm:
    #     ...
    sm_model.to(device)
    choose_a = np.random.uniform(-1, 1, size=10)
    train_SAS, test_SAS, choose_a, sele_list = collect_dyna_sm_data(env,
                                                                    sm_model=sm_model,
                                                                    step_num=NUM_EACH_CYCLE,
                                                                    use_policy=0,
                                                                    choose_a=choose_a)
    train_data = SAS_data(SAS_data=train_SAS)
    test_data = SAS_data(SAS_data=test_SAS)

    log_valid_loss = []

    for epoch_i in range(num_epochs):
        sm_train_valid_loss, sm_model = train_dyna_sm(sm_model, train_data, test_data)

        # Collect one epoch data.
        train_SAS, test_SAS, choose_a, sub_sele_list = collect_dyna_sm_data(env,
                                                                            sm_model=sm_model,
                                                                            train_data=train_data,
                                                                            test_data=test_data,
                                                                            step_num=NUM_EACH_CYCLE,
                                                                            use_policy=1,
                                                                            choose_a=choose_a)
        # Add new data to Data Class
        train_data.add_data(train_SAS)
        test_data.add_data(test_SAS)

        log_valid_loss.append(sm_train_valid_loss)

        # Save dataset.

    PATH = data_save_PTH + "/model_%d" % num_epochs
    torch.save(sm_model.state_dict(), PATH)
    np.savetxt(data_save_PTH + "sm_valid_loss.csv", np.asarray(log_valid_loss))

    S0, A0, NS0, D0 = train_data.all_S, train_data.all_A, train_data.all_NS, train_data.all_DONE
    S1, A1, NS1, D1 = test_data.all_S, test_data.all_A, test_data.all_NS, test_data.all_DONE
    D0 = np.expand_dims(D0, axis=1)
    D1 = np.expand_dims(D1, axis=1)

    data_save0 = np.hstack((S0, A0, NS0, D0))
    data_save1 = np.hstack((S1, A1, NS1, D1))
    data_save = np.vstack((data_save0, data_save1))

    np.save(data_save_PTH + f'sasf_random_total{num_epochs * NUM_EACH_CYCLE}.npy', data_save)








if __name__ == "__main__":


    # Data collection
    mode = 0
    # Run a single robot
    if mode == 0:
        taskID = 9
        print('Task:', taskID)
        data_reward_design = 'loop300'
        num_robots_per_task = 200

        Train = 2
        # p.connect(p.GUI) if not Train else p.connect(p.DIRECT)
        p.connect(p.DIRECT)
        para_config = np.loadtxt('../meta_data/para_config.csv')
        all_robot_test = []
        initial_para = para_config[:,0]
        para_range = para_config[:,1:]
        done_times_log = []
        for robotid in range(taskID*num_robots_per_task, (taskID+1)*num_robots_per_task):

            random.seed(2022)
            np.random.seed(2022)

            robot_name = robot_list[robotid]
            if robot_name in kill_list:
                all_robot_test.append(0)
                print(robot_name,'exist')
                continue
            print(robotid,robot_name)

            initial_joints_angle = np.loadtxt(URDF_PTH + "%s/%s.txt" % (robot_name, robot_name))
            initial_joints_angle = initial_joints_angle[0] if len(initial_joints_angle.shape) == 2 else initial_joints_angle
            # Gait parameters and sans data path:
            log_pth = "../meta_data/%s/%s/" % (data_reward_design, robot_name)
            os.makedirs(log_pth, exist_ok=True)

            if Train == 2:
                env = meta_sm_Env(initial_joints_angle, urdf_path=URDF_PTH + "%s/%s.urdf" % (robot_name, robot_name))
                max_train_step = 300
                env.sleep_time = 0
                obs = env.reset()
                step_times = 0
                r_record = -np.inf
                SANS_data = []
                save_action = []
                done_time = 0
                while 1:
                    all_rewards = []
                    action_list = []
                    norm_space = np.random.sample(len(initial_para))
                    action = norm_space * (para_range[:,1]-para_range[:,0]) + para_range[:,0]

                    action_list.append(action)

                    r = 0
                    for i in range(6):
                        step_times += 1
                        next_obs, _, done, _ = env.step(action)
                        sans = np.hstack((obs, action, next_obs))
                        SANS_data.append(sans)
                        obs = next_obs

                        # stable method
                        r += 3 * obs[1] - abs(obs[0]) - abs(obs[2])

                        if done:
                            obs = env.resetBase()
                            done_time += 1
                            break

                    pos, ori = env.robot_location()
                    # 2. forward_1 method
                    # r = pos[1] - abs(pos[0]) - abs(pos[2]) * 0.4
                    # 2. stable method

                    all_rewards.append(r)

                    if r > r_record:
                        r_record = np.copy(r)
                        save_action = np.copy(action)
                    if step_times >= max_train_step:
                        break
                np.savetxt(log_pth + "gait_step%d.csv" % max_train_step, np.asarray(save_action))
                print("step count:", step_times, "r:", r_record)
                done_times_log.append(done_time)
                np.savetxt(log_pth + "sans_%d.csv" % max_train_step, np.asarray(SANS_data))
                # p.disconnect()


            else:
                env = meta_sm_Env(initial_joints_angle, urdf_path=URDF_PTH + "%s/%s.urdf" % (robot_name, robot_name))
                action = np.loadtxt(log_pth + '/gait_step300.csv')

                env.sleep_time = 0
                obs = env.reset()

                for epoch_run in range(1):
                    obs = env.resetBase()
                    r_log = 0
                    for i in range(6):
                        next_obs, r, done, _ = env.step(action)
                        r_log += r
                        if done:
                            break
                    all_robot_test.append(r_log)
        np.savetxt('../meta_data/test_logger2000_%d.csv'%taskID, np.asarray(all_robot_test))
        # np.savetxt('../meta_data/done_logger2000_%d.csv'%taskID, np.asarray(done_times_log))
