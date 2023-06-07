import os
import random
from random import shuffle
from datetime import datetime
from pos_encoder import *
import wandb
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model import *

RAND_SEED = 43
torch.manual_seed(RAND_SEED)
np.random.seed(RAND_SEED)
random.seed(RAND_SEED)


def unique_leg_conf_idx(robot_names):
    leg_conf_count = dict()
    for robot_name in robot_names:
        name_code = list(map(int, robot_name.split('_')))
        leg_code = (name_code[0], name_code[4], name_code[8], name_code[12])
        if leg_code in leg_conf_count:
            leg_conf_count[leg_code] += 1
        else:
            leg_conf_count[leg_code] = 1
    return leg_conf_count


def joint_range(tag, robot_names):
    all_joint = []
    for robot_name in robot_names:
        name_code = list(map(int, robot_name.split('_')))
        joint_value = np.array(name_code[1:4] + name_code[5:8])
        all_joint.append(joint_value)
    all_joint = np.vstack(all_joint)
    for i in range(6):
        print(tag, i, all_joint[:, i].min(), all_joint[:, i].max())


class SASFDataset(Dataset):
    def __init__(self, data_pth,
                 robot_names,
                 leg2idx,
                 sign_size,
                 max_sample_size,
                 all_sign_flag,
                 torch_device,
                 choose_10steps_input,
                 idx_sample_flag,
                 obs_noise):
        super(SASFDataset, self).__init__()

        self.idx_sample_flag = idx_sample_flag
        self.obs_noise = obs_noise
        self.max_sample_size = max_sample_size
        self.robot_names = robot_names
        self.robot2dynamic = dict()
        self.leg2idx = leg2idx
        self.robot_leg_conf = dict()
        self.robot_joint_conf = dict()
        self.all_sign_loaded = all_sign_flag
        self.device = torch_device
        self.choose_10steps_input = choose_10steps_input
        self.load_data(data_pth, sign_size)

    def __len__(self):
        return len(self.robot_names)

    def __getitem__(self, index):
        robot_name = self.robot_names[index]
        robot_dynamic = self.robot2dynamic[robot_name]
        if self.all_sign_loaded:
            sampled_robot_dynamics = robot_dynamic
            sample_size = len(robot_dynamic)

        else:
            if self.idx_sample_flag == -1:
                random_id = np.random.randint(0, 11 - self.max_sample_size)
                start_point, end_point = random_id * 160, (random_id + self.max_sample_size) * 160
                sampled_robot_dynamics = robot_dynamic[start_point:end_point]
            else:
                sampled_robot_dynamics = robot_dynamic[self.idx_sample_flag* 160:
                                                       (self.idx_sample_flag + self.max_sample_size)* 160]
            # random_id = np.random.choice(10, self.max_sample_size, replace=False)
            # sampled_robot_dynamics = []
            # for rand_id in range(self.max_sample_size):
            #     r_i = random_id[rand_id]
            #     start_point, end_point = r_i * 160, (r_i + 1) * 160
            #     part_data = np.copy(robot_dynamic[start_point:end_point])
            #     sampled_robot_dynamics.append(part_data)
            # sampled_robot_dynamics = np.concatenate(sampled_robot_dynamics)

            # Add noise
            sampled_robot_dynamics[:, 12:] = np.random.normal(sampled_robot_dynamics[:, 12:], self.obs_noise)

            # sample_size = np.random.randint(1, self.max_sample_size + 1)
            # sample_idx = np.random.choice(robot_dynamic.shape[0], sample_size, replace=True)
            # sampled_robot_dynamics = robot_dynamic[sample_idx]

        leg_conf_index = self.leg2idx[self.robot_leg_conf[robot_name]]
        joint_angle_conf = self.robot_joint_conf[robot_name]

        return sampled_robot_dynamics, leg_conf_index, joint_angle_conf, self.max_sample_size

    def load_data(self, data_pth, sign_size):
        for robot_name in tqdm(self.robot_names, desc="Loading Data"):

            # 10 step action + next state 12 + 6 + 12
            dynamic_data = np.load(data_pth[robot_name] + '/sans_%d_0_V2.npy' % (sign_size)).astype(
                dtype=np.float32)
            dyna_shape = dynamic_data.shape
            dynamic_data = dynamic_data.reshape((dyna_shape[0] * dyna_shape[1], dyna_shape[2]))
            # dynamic_data = np.flatten(dynamic_data, start_dim=1, end_dim=2)

            self.robot2dynamic[robot_name] = dynamic_data  # [:self.max_sample_size]

            name_code = list(map(int, robot_name.split('_')))
            if name_code[4] < name_code[0]:
                name_code[:4], name_code[4:8] = name_code[4:8], name_code[:4]
                name_code[8:12], name_code[12:16] = name_code[12:16], name_code[8:12]

            self.robot_leg_conf[robot_name] = (name_code[0], name_code[4], name_code[8], name_code[12])
            self.robot_joint_conf[robot_name] = np.array(name_code[1:4] + name_code[5:8])

    def collate(self, batch):
        sorted_batch = sorted(batch, key=lambda x: x[3], reverse=True)
        dynamics_list = []
        leg_index_list = []
        joint_angle_list = []
        steps_list = []
        length_list = []
        for b in sorted_batch:
            dynamics, leg_index, joint_angle, length = b
            padded_dynamics = np.zeros((self.max_sample_size, 28), dtype=np.float32)
            padded_dynamics[:dynamics.shape[0]] = dynamics
            dynamics_list.append(torch.from_numpy(padded_dynamics))
            leg_index_list.append(torch.tensor(leg_index))
            joint_angle_list.append(torch.from_numpy(joint_angle))
            steps_list.append(torch.tensor(length))
            length_list.append(torch.tensor(dynamics.shape[0]))
        dynamics_torch = pad_sequence(dynamics_list, batch_first=True)
        leg_index_torch = torch.hstack(leg_index_list)
        joint_angle_torch = torch.vstack(joint_angle_list)
        steps_torch = torch.hstack(steps_list)
        length_torch = torch.hstack(length_list)

        return dynamics_torch, leg_index_torch, joint_angle_torch, steps_torch, length_torch


def train():
    dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'
    data_robot_names = open('../data/Jun6_robot_name_200115.txt').read().strip().split('\n')

    pretrained_flag = True
    pretrained = '../data/logger_morning-wind-139/epoch19-acc0.4538'

    use_wandb = True

    wandb.init(project="meta_id_dyna", entity="robotics")  #,mode="disabled"
    config = wandb.config
    config.robot_num = len(data_robot_names)
    config.learning_rate = 0.001
    config.loss_alpha = 0.25
    config.dropout = 0.0
    config.mlp_hidden_dim = 512
    config.MLSTM_hidden_dim = 512
    config.weight_decay = 1e-6
    config.max_sample_size = 1        # 16 sub steps/ step; 10 steps/epoch
    config.choose_10steps_input = True
    config.all_sign_flag = False
    config.obs_noise = 0.0
    max_sample_size = config.max_sample_size
    config.task = 3
    running_name = wandb.run.name
    config.batch_size = 128
    config.pos_encoder = False
    config.torch_device = "cuda:0"
    config.baseline_id = 1

    log_dir = "../data/logger_%s/" % (running_name)
    config.log_dir = log_dir
    config.pre_trained = pretrained_flag
    os.makedirs(log_dir, exist_ok=True)
    PosEnc = PositionalEncoder(d_input=28, n_freqs=5)

    num_epochs = 10000
    num_worker = 5
    sign_size = 100

    robot_paths = dict()
    for rn in data_robot_names:
        rp = dataset_root + 'sign_data/%s' % rn
        robot_paths[rn] = rp

    # unique_leg_count = unique_leg_conf_idx(label_robot_names)
    # joint_range('All joint', robot_names)

    # idx2leg = list(unique_leg_count.keys())
    # np.savetxt('../data/leg_labels.csv', np.asarray(idx2leg), fmt="%i")
    # leg2idx = {leg: idx for idx, leg in enumerate(idx2leg)}
    idx2leg = np.loadtxt('../data/leg_labels.csv')
    leg2idx = dict()
    for i in range(len(idx2leg)):
        leg2idx[tuple(idx2leg[i])] = i

    # shuffle(robot_names)
    split_idx = int(len(data_robot_names) * 0.8)

    train_robot_names = data_robot_names[:split_idx]
    valid_robot_names = data_robot_names[split_idx:]

    train_unique_leg_count = unique_leg_conf_idx(train_robot_names)
    valid_unique_leg_count = unique_leg_conf_idx(valid_robot_names)

    print("Num of train unique conf:", len(train_unique_leg_count))
    print("Num of valid unique conf:", len(valid_unique_leg_count))

    print("Num of Train Robots:", len(train_robot_names))
    print("Num of Valid Robots:", len(valid_robot_names))

    train_dataset = SASFDataset(robot_paths,
                                train_robot_names,
                                leg2idx,
                                sign_size=sign_size,
                                max_sample_size=max_sample_size,
                                all_sign_flag=config.all_sign_flag,
                                torch_device=config.torch_device,
                                choose_10steps_input=config.choose_10steps_input,
                                obs_noise=config.obs_noise,
                                idx_sample_flag=-1
                                )
    valid_dataset = SASFDataset(robot_paths,
                                valid_robot_names,
                                leg2idx,
                                sign_size=sign_size,
                                max_sample_size=max_sample_size,
                                all_sign_flag=config.all_sign_flag,
                                torch_device=config.torch_device,
                                choose_10steps_input=config.choose_10steps_input,
                                obs_noise=config.obs_noise,
                                idx_sample_flag=-1
                                )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=num_worker)  # , collate_fn=train_dataset.collate
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False,
                              num_workers=num_worker)  # , collate_fn=valid_dataset.collate

    # Setup model
    model = PredConf(state_dim=30,
                     do=config.dropout,
                     MLSTM_hidden_dim=config.MLSTM_hidden_dim,
                     mlp_hidden_dim=config.mlp_hidden_dim,
                     single_objective=config.task,
                     device=config.torch_device,
                     baseline_id = config.baseline_id)

    if pretrained_flag == True:
        pretrained_model_dict = torch.load(pretrained)
        model_dict = model.state_dict()
        partial_state_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
        model_dict.update(partial_state_dict)
        model.load_state_dict(model_dict)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    model = model.to(config.torch_device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=0.5,
                                                           patience=40,
                                                           verbose=True)

    best_valid_acc = 0

    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_running_joint_loss = 0.0
        train_running_leg_loss = 0.0

        train_correct_leg = 0.0
        train_correct_joint = 0.0
        train_joint_sample_num = 0.0
        train_leg_sample_num = 0.0

        train_b_num = 0
        for batch in tqdm(train_loader, desc="Training"):
            model.train()

            memory, gt_leg_cfg, gt_joint_cfg, length = batch
            if config.pos_encoder:
                memory = PosEnc(memory)
            memory = memory.to(config.torch_device)
            gt_leg_cfg = gt_leg_cfg.to(config.torch_device)
            gt_joint_cfg = gt_joint_cfg.to(config.torch_device)
            optimizer.zero_grad()
            # memory: batch x length x channels
            pred_leg_cfg, pred_joint_cfg = model(memory, length)

            leg_loss = criterion1(pred_leg_cfg, gt_leg_cfg)
            pred_joint_cfg = torch.cat(pred_joint_cfg)
            gt_joint_cfg = gt_joint_cfg.T.flatten()
            joint_loss = criterion2(pred_joint_cfg, gt_joint_cfg)
            if config.loss_alpha == 2:
                loss = leg_loss + joint_loss
            else:
                loss = (config.loss_alpha * leg_loss + (1 - config.loss_alpha) * joint_loss)

            loss.backward()
            optimizer.step()

            train_running_leg_loss += leg_loss.item()
            train_running_joint_loss += joint_loss.item()
            train_running_loss += loss.item()

            pred_leg_cfg_id = torch.argmax(pred_leg_cfg, dim=1)
            pred_joint_cfg_id = torch.argmax(pred_joint_cfg, dim=1)
            train_correct_leg += (pred_leg_cfg_id == gt_leg_cfg).sum().item()
            train_correct_joint += (pred_joint_cfg_id == gt_joint_cfg).sum().item()
            train_joint_sample_num += len(gt_joint_cfg)
            train_leg_sample_num += len(gt_leg_cfg)
            train_b_num += 1

        train_running_leg_loss /= train_b_num
        train_running_joint_loss /= train_b_num
        train_running_loss /= train_b_num

        #################################################
        valid_running_loss = 0.0
        valid_running_joint_loss = 0.0
        valid_running_leg_loss = 0.0

        valid_correct_leg = 0.0
        valid_correct_joint = 0.0
        valid_joint_sample_num = 0.0
        valid_leg_sample_num = 0.0
        model.eval()
        valid_b_num = 0
        for batch in tqdm(valid_loader, desc="Validing"):
            memory, gt_leg_cfg, gt_joint_cfg, length = batch
            if config.pos_encoder:
                memory = PosEnc(memory)
            memory = memory.to(config.torch_device)
            gt_leg_cfg = gt_leg_cfg.to(config.torch_device)
            gt_joint_cfg = gt_joint_cfg.to(config.torch_device)

            with torch.no_grad():
                pred_leg_cfg, pred_joint_cfg = model(memory, length)
                leg_loss = criterion1(pred_leg_cfg, gt_leg_cfg)
                pred_joint_cfg = torch.cat(pred_joint_cfg)
                gt_joint_cfg = gt_joint_cfg.T.flatten()

                joint_loss = criterion2(pred_joint_cfg, gt_joint_cfg)
                if config.loss_alpha == 2:
                    loss = leg_loss + joint_loss
                else:
                    loss = (config.loss_alpha * leg_loss + (1 - config.loss_alpha) * joint_loss)

                valid_running_leg_loss += leg_loss.item()
                valid_running_joint_loss += joint_loss.item()
                valid_running_loss += loss.item()

                pred_leg_cfg_id = torch.argmax(pred_leg_cfg, dim=1)
                pred_joint_cfg_id = torch.argmax(pred_joint_cfg, dim=1)
                valid_correct_leg += (pred_leg_cfg_id == gt_leg_cfg).sum().item()
                valid_correct_joint += (pred_joint_cfg_id == gt_joint_cfg).sum().item()
                valid_joint_sample_num += len(gt_joint_cfg)
                valid_leg_sample_num += len(gt_leg_cfg)
            valid_b_num += 1

        scheduler.step(valid_running_loss)
        train_acc = (train_correct_joint + train_correct_leg) / (train_joint_sample_num + train_leg_sample_num)
        train_joint_acc = train_correct_joint / train_joint_sample_num
        train_leg_acc = train_correct_leg / train_leg_sample_num

        valid_acc = (valid_correct_joint + valid_correct_leg) / (valid_joint_sample_num + valid_leg_sample_num)
        valid_joint_acc = valid_correct_joint / valid_joint_sample_num
        valid_leg_acc = valid_correct_leg / valid_leg_sample_num
        valid_running_leg_loss /= valid_b_num
        valid_running_joint_loss /= valid_b_num
        valid_running_loss /= valid_b_num

        # Computing Early Stopping
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc

            model_name = f"epoch{epoch + 1}-acc{best_valid_acc:.4f}"
            torch.save(model.state_dict(), os.path.join(log_dir, model_name))

        current_lr = optimizer.param_groups[0]['lr']
        # Logging
        print(
            f"\n[{datetime.now()}] Model Params: {num_params} Epoch [{epoch + 1} / {num_epochs}] Best Valid Avg Joint Acc {best_valid_acc:.4f}")
        print(
            f"Train Total Loss: {train_running_loss:.4f} Leg Loss: {train_running_leg_loss:.4f} Joint Loss: {train_running_joint_loss:.4f} Leg Acc: {train_leg_acc:.4f} Joint Acc: {train_joint_acc:.4f} Overall Acc: {train_acc:.4f}")
        print(
            f"Valid Total Loss: {valid_running_loss:.4f} Leg Loss: {valid_running_leg_loss:.4f} Joint Loss: {valid_running_joint_loss:.4f} Leg Acc: {valid_leg_acc:.4f} Joint Acc: {valid_joint_acc:.4f} Overall Acc: {valid_acc:.4f}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_running_loss,
                'valid_loss': valid_running_loss,
                'train_leg_loss': train_running_leg_loss,
                'valid_leg_loss': valid_running_leg_loss,
                'train_joint_loss': train_running_joint_loss,
                'valid_joint_loss': valid_running_joint_loss,
                'train_leg_acc': train_leg_acc,
                'valid_leg_acc': valid_leg_acc,
                'train_joint_acc': train_joint_acc,
                'valid_joint_acc': valid_joint_acc,
                'train_overall_acc': train_acc,
                'valid_overall_acc': valid_acc,
                'best_valid_avg_joint_acc': best_valid_acc,
                'learning_rate': current_lr
            })

        # if early_stop_offset == early_stop_interval:
        #     break


if __name__ == '__main__':
    # wandb.agent('vcw5rra5', function=main, count=5)
    train()
