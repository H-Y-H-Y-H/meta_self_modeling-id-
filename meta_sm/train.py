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
    def __init__(self, robot_paths, robot_names, leg2idx,sign_size, max_sample_size,all_sign_flag):
        super(SASFDataset, self).__init__()
        self.max_sample_size = max_sample_size
        self.robot_names = robot_names
        self.robot2dynamic = dict()
        self.leg2idx = leg2idx
        self.robot_leg_conf = dict()
        self.robot_joint_conf = dict()
        self.load_data(robot_paths,sign_size)
        self.all_sign_loaded = all_sign_flag

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
        return sampled_robot_dynamics, leg_conf_index, joint_angle_conf, sample_size

    def load_data(self, robot_paths,sign_size):
        for robot_name in tqdm(self.robot_names, desc="Loading Data"):
            dynamic_data = np.loadtxt(robot_paths[robot_name]+'/sans_%d_0.csv'%(self.max_sample_size-1)).astype(dtype=np.float32)
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
    log_dir = "../data/logger_256/"
    os.makedirs(log_dir, exist_ok=True)

    use_wandb = True
    if use_wandb:
        wandb.init(project="meta_encoder", entity="robotics")
        config = wandb.config
        config.learning_rate = 0.01
        config.loss_alpha = 0.75
        config.dropout = 0
        config.mlp_hidden_dim = 256
        config.MLSTM_hidden_dim = 256
        config.weight_decay = 1e-6
        config.max_sample_size = 201
        config.encoder_type = 0
        max_sample_size = config.max_sample_size
    else:
        max_sample_size = 201

    num_epochs = 10000
    batch_size = 32
    use_gpu = True
    torch_device = "cuda:0"
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
    robot_names = open('../data/f_robot_names35k.txt').read().strip().split('\n')
    for rn in robot_names:
        rp = dataset_root + 'data/robot_sign_data_2/%s'%rn
        robot_paths[rn] = rp

    unique_leg_count = unique_leg_conf_idx(robot_names)
    # joint_range('All joint', robot_names)

    idx2leg = list(unique_leg_count.keys())
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

    train_dataset = SASFDataset(robot_paths, train_robot_names, leg2idx,sign_size = sign_size, max_sample_size=max_sample_size,all_sign_flag=True)
    valid_dataset = SASFDataset(robot_paths, valid_robot_names, leg2idx,sign_size = sign_size, max_sample_size=max_sample_size,all_sign_flag=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker, collate_fn=train_dataset.collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, collate_fn=valid_dataset.collate)

    # Setup model
    model = PredConf(state_dim=28, do=config.dropout,MLSTM_hidden_dim=config.MLSTM_hidden_dim, mlp_hidden_dim=config.mlp_hidden_dim, encoder_type = config.encoder_type)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    device = torch.device(torch_device if use_gpu and torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training
    train_loss_avg = [0]
    valid_loss_avg = [0]
    train_leg_loss_avg = [0]
    valid_leg_loss_avg = [0]
    valid_joint_loss_avg = [0]
    train_joint_loss_avg = [0]

    best_valid_avg_joint_acc = 0
    early_stop_offset = 0

    for epoch in range(num_epochs):
        model.train()
        train_num_batches = 0
        train_leg_ground_truth = []
        train_joint_conf_ground_truth = []
        train_joint_conf_pred = []
        train_leg_pred_label = []
        train_steps = []
        for batch in tqdm(train_loader, desc="Training"):
            memory, leg_conf_id, joint_conf, steps, length = batch
            # memory: batch x length x channels
            train_leg_ground_truth.extend(list(leg_conf_id.numpy()))
            train_joint_conf_ground_truth.append(joint_conf.detach().cpu().numpy())
            train_steps.extend(list(steps.numpy()))

            memory = memory.to(device)
            leg_conf_id = leg_conf_id.to(device)
            joint_conf = joint_conf.to(device)

            leg_logit, pred_joint_confs = model(memory, length)
            leg_loss = F.cross_entropy(leg_logit, leg_conf_id)

            joint_losses = []
            for j in range(6):
                pred_joint_logit = pred_joint_confs[j]
                joint_label = joint_conf[:, j]
                joint_losses.append(F.cross_entropy(pred_joint_logit, joint_label))

            loss = config.loss_alpha * (sum(joint_losses) / 6) + (1 - config.loss_alpha) * leg_loss

            # loss = loss_alpha * leg_loss + (1 - loss_alpha) * joint_loss
            # loss = leg_loss + joint_loss

            with torch.no_grad():
                train_leg_pred_label.extend(list(torch.argmax(leg_logit, dim=1).detach().cpu().numpy()))
                train_joint_conf_pred.append(np.hstack([torch.argmax(pred_joint_confs[i], dim=1).detach().cpu().numpy()[:, None] for i in range(6)]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_leg_loss_avg[-1] += leg_loss.item()
            train_loss_avg[-1] += loss.item()
            train_joint_loss_avg[-1] += (sum(joint_losses) / 6).item()
            train_num_batches += 1

        model.eval()
        val_num_batches = 0
        leg_ground_truth = []
        leg_pred_label = []
        joint_conf_ground_truth = []
        joint_conf_pred = []
        valid_steps = []
        for batch in tqdm(valid_loader, desc="Validing"):
            memory, leg_conf_id, joint_conf, steps, length = batch

            leg_ground_truth.extend(list(leg_conf_id.numpy()))
            joint_conf_ground_truth.append(joint_conf.detach().cpu().numpy())
            valid_steps.extend(list(steps.numpy()))

            memory = memory.to(device)
            leg_conf_id = leg_conf_id.to(device)
            joint_conf = joint_conf.to(device)

            with torch.no_grad():
                leg_logit, pred_joint_confs = model(memory, length)
                leg_loss = F.cross_entropy(leg_logit, leg_conf_id)
                joint_losses = []
                for j in range(6):
                    pred_joint_logit = pred_joint_confs[j]
                    joint_label = joint_conf[:, j]
                    joint_losses.append(F.cross_entropy(pred_joint_logit, joint_label))

                loss = config.loss_alpha * (sum(joint_losses) / 6) + (1 - config.loss_alpha) * leg_loss

                leg_pred_label.extend(list(torch.argmax(leg_logit, dim=1).detach().cpu().numpy()))
                joint_conf_pred.append(np.hstack([torch.argmax(pred_joint_confs[i], dim=1).detach().cpu().numpy()[:, None] for i in range(6)]))

                # joint_conf_pred.extend(torch.round(pred_joint_conf).detach().cpu().numpy())

            valid_leg_loss_avg[-1] += leg_loss.item()
            valid_joint_loss_avg[-1] += (sum(joint_losses) / 6).item()
            valid_loss_avg[-1] += loss.item()
            val_num_batches += 1

        # Computing Evaluation Metrics
        train_loss_avg[-1] /= train_num_batches
        valid_loss_avg[-1] /= val_num_batches
        train_leg_loss_avg[-1] /= train_num_batches
        valid_leg_loss_avg[-1] /= val_num_batches
        train_joint_loss_avg[-1] /= train_num_batches
        valid_joint_loss_avg[-1] /= val_num_batches

        joint_conf_ground_truth = np.vstack(joint_conf_ground_truth)
        joint_conf_pred = np.vstack(joint_conf_pred)

        train_joint_conf_ground_truth = np.vstack(train_joint_conf_ground_truth)
        train_joint_conf_pred = np.vstack(train_joint_conf_pred)

        train_leg_correct = 0
        train_joint_correct = 0
        train_all_correct = 0
        train_total = train_joint_conf_ground_truth.shape[0]
        for i in range(train_total):
            if all(train_joint_conf_ground_truth[i, :] == train_joint_conf_pred[i, :]):
                train_joint_correct += 1
            if train_leg_ground_truth[i] == train_leg_pred_label[i]:
                train_leg_correct += 1
            if train_leg_ground_truth[i] == train_leg_pred_label[i] and all(
                    train_joint_conf_ground_truth[i, :] == train_joint_conf_pred[i, :]):
                train_all_correct += 1

        train_total = train_joint_conf_ground_truth.shape[0]
        train_steps_joint_part_correct = np.zeros((6, config.max_sample_size))
        train_joint_part_correct = [0] * 6
        for i in range(train_total):
            for j in range(6):
                if train_joint_conf_pred[i][j] == train_joint_conf_ground_truth[i][j]:
                    train_joint_part_correct[j] += 1
                    train_steps_joint_part_correct[j, train_steps[i]-1] += 1

        train_leg_acc = train_leg_correct / train_total
        train_joint_acc = train_joint_correct / train_total
        train_overall_acc = train_all_correct / train_total

        leg_correct = 0
        joint_correct = 0
        all_correct = 0
        total = joint_conf_ground_truth.shape[0]
        for i in range(total):
            if all(joint_conf_ground_truth[i, :] == joint_conf_pred[i, :]):
                joint_correct += 1
            if leg_ground_truth[i] == leg_pred_label[i]:
                leg_correct += 1
            if leg_ground_truth[i] == leg_pred_label[i] and all(joint_conf_ground_truth[i, :] == joint_conf_pred[i, :]):
                all_correct += 1

        joint_part_correct = [0] * 6
        valid_steps_joint_part_correct = np.zeros((6, config.max_sample_size))
        for i in range(total):
            for j in range(6):
                if joint_conf_pred[i][j] == joint_conf_ground_truth[i][j]:
                    joint_part_correct[j] += 1
                    valid_steps_joint_part_correct[j, valid_steps[i]-1] += 1

        valid_leg_acc = leg_correct / total
        valid_joint_acc = joint_correct / total
        valid_overall_acc = all_correct / total

        train_avg_joint_acc = sum(train_joint_part_correct[j] / train_total for j in range(6)) / 6
        valid_avg_joint_acc = sum(joint_part_correct[j] / total for j in range(6)) / 6

        train_steps_count = [train_steps.count(i) for i in range(1, config.max_sample_size + 1)]
        valid_steps_count = [valid_steps.count(i) for i in range(1, config.max_sample_size + 1)]

        train_steps_joint_part_correct_sum = train_steps_joint_part_correct.sum(axis=0)
        valid_steps_joint_part_correct_sum = valid_steps_joint_part_correct.sum(axis=0)

        # Computing Early Stopping
        if valid_avg_joint_acc >= best_valid_avg_joint_acc:
            best_valid_avg_joint_acc = valid_avg_joint_acc
            if epoch > 500:
                model_name = f"epoch{epoch+1}-acc{valid_avg_joint_acc:.4f}"
                torch.save(model.state_dict(), os.path.join(log_dir, model_name))
            # early_stop_offset = 0
        # else:
            # early_stop_offset += 1

        # print(
        #     'Sample Size: %d Epoch [%d / %d] train loss: %.4f valid loss: %.4f valid leg loss: %.4f valid joint loss: %.4f train joint loss: %.4f leg acc: %.4f joint acc: %.4f overall acc: %.4f' % (
        #         MEMORY_SAMPLE_SIZE, epoch + 1, num_epochs, train_loss_avg[-1], valid_loss_avg[-1],
        #         valid_leg_loss_avg[-1], valid_joint_loss_avg[-1], train_joint_loss_avg[-1], leg_acc, joint_acc,
        #         overall_acc))

        # Logging
        print(f"\n[{datetime.now()}] Model Params: {num_params} Epoch [{epoch+1} / {num_epochs}] Best Valid Avg Joint Acc {best_valid_avg_joint_acc:.4f}")
        print(f"Train Total Loss: {train_loss_avg[-1]:.4f} Leg Loss: {train_leg_loss_avg[-1]:.4f} Joint Loss: {train_joint_loss_avg[-1]:.4f} Leg Acc: {train_leg_acc:.4f} Joint Acc: {train_joint_acc:.4f} Overall Acc: {train_overall_acc:.4f} Avg Joint Acc: {train_avg_joint_acc:.3f}")
        print(f"Valid Total Loss: {valid_loss_avg[-1]:.4f} Leg Loss: {valid_leg_loss_avg[-1]:.4f} Joint Loss: {valid_joint_loss_avg[-1]:.4f} Leg Acc: {valid_leg_acc:.4f} Joint Acc: {valid_joint_acc:.4f} Overall Acc: {valid_overall_acc:.4f} Avg Joint Acc: {valid_avg_joint_acc:.3f}")

        print("Train Joints Acc", end=" ")
        for j in range(6):
            print(f'J{j}: {train_joint_part_correct[j] / train_total:.4f}', end=' ')
        print()

        print("Valid Joints Acc", end=" ")
        for j in range(6):
            print(f'J{j}: {joint_part_correct[j] / total:.4f}', end=' ')
        print()

        # print("Train Steps Acc", end=" ")
        # for j in range(config.max_sample_size):
        #     print(f'S{j}: {train_steps_joint_part_correct_sum[j] / train_steps_count[j]:.4f}', end=' ')
        # print()
        #
        # print("Valid Steps Acc", end=" ")
        # for j in range(config.max_sample_size):
        #     print(f'S{j}: {valid_steps_joint_part_correct_sum[j] / valid_steps_count[j]:.4f}', end=' ')
        # print('\n')



        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss_avg[-1],
                'valid_loss': valid_loss_avg[-1],
                'train_leg_loss': train_leg_loss_avg[-1],
                'valid_leg_loss': valid_leg_loss_avg[-1],
                'train_joint_loss': train_joint_loss_avg[-1],
                'valid_joint_loss': valid_joint_loss_avg[-1],
                'train_leg_acc': train_leg_acc,
                'valid_leg_acc': valid_leg_acc,
                'train_joint_acc': train_joint_acc,
                'valid_joint_acc': valid_joint_acc,
                'train_overall_acc': train_overall_acc,
                'valid_overall_acc': valid_overall_acc,
                'train_avg_joint_acc': train_avg_joint_acc,
                'valid_avg_joint_acc': valid_avg_joint_acc,
                'best_valid_avg_joint_acc': best_valid_avg_joint_acc
            })

        # if early_stop_offset == early_stop_interval:
        #     break

if __name__ == '__main__':
    # wandb.agent('vcw5rra5', function=main, count=5)
    train()