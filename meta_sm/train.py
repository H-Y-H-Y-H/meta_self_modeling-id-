import os
import random
from random import shuffle
from datetime import datetime

import wandb
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from model import *

RAND_SEED = 42
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
    def __init__(self, robot_paths, robot_names, leg2idx, sign_size, max_sample_size, all_sign_flag,torch_device):
        super(SASFDataset, self).__init__()
        self.max_sample_size = max_sample_size
        self.robot_names = robot_names
        self.robot2dynamic = dict()
        self.leg2idx = leg2idx
        self.robot_leg_conf = dict()
        self.robot_joint_conf = dict()
        self.load_data(robot_paths, sign_size)
        self.all_sign_loaded = all_sign_flag
        self.device = torch_device
    def __len__(self):
        return len(self.robot_names)

    def __getitem__(self, index):
        robot_name = self.robot_names[index]
        robot_dynamic = self.robot2dynamic[robot_name]
        if self.all_sign_loaded:
            sampled_robot_dynamics = robot_dynamic
            sample_size = len(robot_dynamic)

        else:
            sample_size = np.random.randint(1, self.max_sample_size + 1)
            sample_idx = np.random.choice(robot_dynamic.shape[0], sample_size, replace=True)
            sampled_robot_dynamics = robot_dynamic[sample_idx]

        leg_conf_index = self.leg2idx[self.robot_leg_conf[robot_name]]
        joint_angle_conf = self.robot_joint_conf[robot_name]

        # sampled_robot_dynamics = torch.from_numpy(sampled_robot_dynamics).to(self.device)
        # leg_conf_index = torch.from_numpy(leg_conf_index).to(self.device)
        # joint_angle_conf = torch.from_numpy(joint_angle_conf).to(self.device)

        return sampled_robot_dynamics, leg_conf_index, joint_angle_conf, sample_size

    def load_data(self, robot_paths, sign_size):
        for robot_name in tqdm(self.robot_names, desc="Loading Data"):
            dynamic_data = np.loadtxt(robot_paths[robot_name] + '/sans_%d_0.csv' % (self.max_sample_size - 1)).astype(
                dtype=np.float32)
            self.robot2dynamic[robot_name] = dynamic_data[:sign_size]

            # dynamic_steps = dynamic_data[..., 1:].reshape(-1, 18, 16, 6).transpose(1, 0, 2, 3).reshape(18, -1, 16)
            # normalized_dynamic_steps = dynamic_steps - dynamic_steps[..., 0, None]
            # normalized_dynamic_steps = dynamic_steps
            # self.robot2dynamic[robot_name] = normalized_dynamic_steps

            name_code = list(map(int, robot_name.split('_')))
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

    pretrained_flag = False
    # pretrained = '../data/logger_128k/epoch566-acc0.3402'
    # pretrained = '../data/logger_128k_0.250000_1024_2/epoch131-acc0.3332'
    pretrained = '../data/logger_128k_256_2/epoch54-acc0.4815'

    use_wandb = True

    wandb.init(project="meta_id", entity="robotics")  #,, mode="disabled"
    config = wandb.config
    config.learning_rate = 0.001
    config.loss_alpha = 0.25
    config.dropout = 0.05
    config.mlp_hidden_dim = 256
    config.MLSTM_hidden_dim = 256
    config.weight_decay = 1e-6
    config.max_sample_size = 201
    config.encoder_type = 0
    max_sample_size = config.max_sample_size
    config.task = 2
    running_name = wandb.run.name

    log_dir = "../data/logger_%s/"%(running_name)
    config.log_dir = log_dir
    config.pre_trained = pretrained_flag
    os.makedirs(log_dir, exist_ok=True)


    num_epochs = 10000
    batch_size = 8
    torch_device = "cuda:1"
    # torch_device = "cpu"

    num_worker = 5
    sign_size = 201
    # early_stop_interval = 500
    # n_dataset = 1000

    # learning_rate = wandb.config.learning_rate if use_wandb else 0.003
    # loss_alpha = wandb.config.loss_alpha if use_wandb else 0.75
    # mlp_encode_type = wandb.config.mlp_encode_type if use_wandb else 1
    # do = wandb.config.dropout if use_wandb else 0.3
    # mlp_hidden_dim = wandb.config.mlp_hidden_dim if use_wandb else 256
    # weight_decay = wandb.config.weight_decay if use_wandb else 0
    # pred_joint_type = wandb.config.pred_joint_type if use_wandb else 0

    robot_paths = dict()
    robot_names = open('../data/f_robot_names128295.txt').read().strip().split('\n')#[:1000]
    for rn in robot_names:
        rp = dataset_root + 'data/robot_sign_data_2/%s' % rn
        robot_paths[rn] = rp

    unique_leg_count = unique_leg_conf_idx(robot_names)
    # joint_range('All joint', robot_names)

    idx2leg = list(unique_leg_count.keys())
    np.savetxt('../data/leg_labels.csv', np.asarray(idx2leg), fmt="%i")

    leg2idx = {leg: idx for idx, leg in enumerate(idx2leg)}

    shuffle(robot_names)
    split_idx = int(len(robot_names) * 0.8)

    train_robot_names = robot_names[:split_idx]
    valid_robot_names = robot_names[split_idx:]
    # joint_range('Train joint', robot_names)
    # joint_range('Valid joint', robot_names)

    train_unique_leg_count = unique_leg_conf_idx(train_robot_names)
    valid_unique_leg_count = unique_leg_conf_idx(valid_robot_names)

    print("Num of unique conf:", len(unique_leg_count))
    print("Num of train unique conf:", len(train_unique_leg_count))
    print("Num of valid unique conf:", len(valid_unique_leg_count))
    # assert len(train_unique_leg_count) == len(unique_leg_count) and len(valid_unique_leg_count) == len(unique_leg_count)

    print("Num of Train Robots:", len(train_robot_names))
    print("Num of Valid Robots:", len(valid_robot_names))

    train_dataset = SASFDataset(robot_paths, train_robot_names, leg2idx, sign_size=sign_size,
                                max_sample_size=max_sample_size, all_sign_flag=True,torch_device=torch_device)
    valid_dataset = SASFDataset(robot_paths, valid_robot_names, leg2idx, sign_size=sign_size,
                                max_sample_size=max_sample_size, all_sign_flag=True,torch_device=torch_device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_worker)  # , collate_fn=train_dataset.collate
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_worker)  # , collate_fn=valid_dataset.collate

    # Setup model
    model = PredConf(state_dim=28,
                     do=config.dropout,
                     MLSTM_hidden_dim=config.MLSTM_hidden_dim,
                     mlp_hidden_dim=config.mlp_hidden_dim,
                     encoder_type=config.encoder_type,
                     single_objective=config.task,
                     device=torch_device)

    if pretrained_flag == True:
        pretrained_model_dict = torch.load(pretrained)
        model_dict = model.state_dict()
        partial_state_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict}
        model_dict.update(partial_state_dict)
        model.load_state_dict(model_dict)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    model = model.to(torch_device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.1,patience=20,verbose=True)

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

            # memory, leg_conf_id, joint_conf, steps, length = batch
            memory, gt_leg_cfg, gt_joint_cfg, length = batch
            memory = memory.to(torch_device)
            gt_leg_cfg = gt_leg_cfg.to(torch_device)
            gt_joint_cfg = gt_joint_cfg.to(torch_device)

            optimizer.zero_grad()

            # memory: batch x length x channels
            pred_leg_cfg, pred_joint_cfg = model(memory, length)

            leg_loss = criterion1(pred_leg_cfg, gt_leg_cfg)
            pred_joint_cfg = torch.cat(pred_joint_cfg)
            gt_joint_cfg = gt_joint_cfg.T.flatten()
            joint_loss = criterion2(pred_joint_cfg, gt_joint_cfg)

            loss = (config.loss_alpha * leg_loss + (1 - config.loss_alpha) * joint_loss)

            loss.backward()
            optimizer.step()

            train_running_leg_loss   += leg_loss.item()
            train_running_joint_loss += joint_loss.item()
            train_running_loss       += loss.item()

            pred_leg_cfg_id = torch.argmax(pred_leg_cfg, dim=1)
            pred_joint_cfg_id = torch.argmax(pred_joint_cfg, dim=1)
            train_correct_leg += (pred_leg_cfg_id == gt_leg_cfg).sum().item()
            train_correct_joint += (pred_joint_cfg_id == gt_joint_cfg).sum().item()
            train_joint_sample_num += len(gt_joint_cfg)
            train_leg_sample_num += len(gt_leg_cfg)
            train_b_num+=1

        train_running_leg_loss   /= train_b_num
        train_running_joint_loss /= train_b_num
        train_running_loss       /= train_b_num

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
            memory = memory.to(torch_device)
            gt_leg_cfg = gt_leg_cfg.to(torch_device)
            gt_joint_cfg = gt_joint_cfg.to(torch_device)

            with torch.no_grad():
                pred_leg_cfg, pred_joint_cfg = model(memory, length)

                leg_loss = criterion1(pred_leg_cfg, gt_leg_cfg)
                pred_joint_cfg = torch.cat(pred_joint_cfg)
                gt_joint_cfg = gt_joint_cfg.T.flatten()
                # print(pred_joint_cfg.shape,gt_joint_cfg.shape)
                joint_loss = criterion2(pred_joint_cfg, gt_joint_cfg)
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
            valid_b_num+=1

        scheduler.step(valid_running_loss)
        train_acc = (train_correct_joint+train_correct_leg)/(train_joint_sample_num+train_leg_sample_num)
        train_joint_acc = train_correct_joint/train_joint_sample_num
        train_leg_acc = train_correct_leg/train_leg_sample_num

        valid_acc = (valid_correct_joint+valid_correct_leg)/(valid_joint_sample_num+valid_leg_sample_num)
        valid_joint_acc = valid_correct_joint/valid_joint_sample_num
        valid_leg_acc = valid_correct_leg/valid_leg_sample_num
        valid_running_leg_loss   /= valid_b_num
        valid_running_joint_loss /= valid_b_num
        valid_running_loss       /= valid_b_num

        # Computing Early Stopping
        if valid_acc >= best_valid_acc:
            best_valid_acc = valid_acc
            if (epoch > 10):
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
