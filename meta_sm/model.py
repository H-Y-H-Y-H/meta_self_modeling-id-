import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class MLSTMfcn(nn.Module):
    def __init__(self, num_features,
                 num_lstm_out=256, num_lstm_layers=2,
                 conv1_nf=256, conv2_nf=512, conv3_nf=256,
                 lstm_drop_p=0, fc_drop_p=0):
        super(MLSTMfcn, self).__init__()

        self.num_features = num_features

        self.num_lstm_out = num_lstm_out
        self.num_lstm_layers = num_lstm_layers

        self.conv1_nf = conv1_nf
        self.conv2_nf = conv2_nf
        self.conv3_nf = conv3_nf

        self.lstm_drop_p = lstm_drop_p
        self.fc_drop_p = fc_drop_p

        self.lstm = nn.LSTM(input_size=self.num_features,
                            hidden_size=self.num_lstm_out,
                            num_layers=self.num_lstm_layers,
                            batch_first=True)

        self.conv1 = nn.Conv1d(self.num_features, self.conv1_nf, 8)
        self.conv2 = nn.Conv1d(self.conv1_nf, self.conv2_nf, 5)
        self.conv3 = nn.Conv1d(self.conv2_nf, self.conv3_nf, 3)

        self.bn1 = nn.BatchNorm1d(self.conv1_nf)
        self.bn2 = nn.BatchNorm1d(self.conv2_nf)
        self.bn3 = nn.BatchNorm1d(self.conv3_nf)

        self.se1 = SELayer(self.conv1_nf)  # ex 128
        self.se2 = SELayer(self.conv2_nf)  # ex 256

        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(self.lstm_drop_p)
        self.convDrop = nn.Dropout(self.fc_drop_p)

        self.global_feat_dim = self.conv3_nf + self.num_lstm_out

        self.output_size = self.global_feat_dim

    def forward(self, x, length):
        # x: batch x seq_len x channels
        # packed_x = pack_padded_sequence(x, length.cpu().numpy(), batch_first=True)

        packed_x_out, (ht, ct) = self.lstm(x)
        x1 = ht[-1]

        x2 = x.transpose(2, 1)
        x2 = self.convDrop(self.relu(self.bn1(self.conv1(x2))))
        x2 = self.se1(x2)
        x2 = self.convDrop(self.relu(self.bn2(self.conv2(x2))))
        x2 = self.se2(x2)
        x2 = self.convDrop(self.relu(self.bn3(self.conv3(x2))))
        x2 = torch.mean(x2, 2)
        x_all = torch.cat((x1, x2), dim=1)

        return x_all


class SequentialNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        return x


class PredConf(nn.Module):
    def __init__(self, state_dim,
                 MLSTM_hidden_dim, single_objective=0,
                 num_class=30, num_joint=12,
                 do=0., mlp_hidden_dim=256,
                 device='cuda:1'):
        super(PredConf, self).__init__()

        self.device = device
        self.signature_encode = MLSTMfcn(state_dim,
                                         num_lstm_out=MLSTM_hidden_dim,
                                         num_lstm_layers=2,
                                         conv1_nf=MLSTM_hidden_dim,
                                         conv2_nf=MLSTM_hidden_dim * 2,
                                         conv3_nf=MLSTM_hidden_dim, )

        self.pred_mlp = nn.Sequential(
            nn.Linear(self.signature_encode.output_size, mlp_hidden_dim),
            nn.Dropout(p=do),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Dropout(p=do),
            nn.BatchNorm1d(mlp_hidden_dim),
            nn.ReLU(),
        )

        self.single_objective = single_objective
        self.pred_leg_head = nn.Linear(mlp_hidden_dim, num_class)
        self.pred_leg_head_0 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 4)
        self.pred_leg_head_1 = nn.Linear(mlp_hidden_dim // 4, num_class)

        self.pred_joint_heads_seq = nn.ModuleList(
            [self.joint_heads_seq(mlp_hidden_dim, num_joint, do) for _ in range(6)])

        self.pred_joint_heads = nn.ModuleList([nn.Linear(mlp_hidden_dim, num_joint) for i in range(6)])

        self.relu = nn.ReLU()

    def joint_heads_seq(self, mlp_hidden_dim, num_joint, do):
        return nn.Sequential(
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Dropout(p=do),
            # nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.Dropout(p=do),
            # nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim//4),
            nn.Dropout(p=do),
            # nn.ReLU(),
            nn.Linear(mlp_hidden_dim//4, num_joint),
        )

    def get_latent(self, memory, length):
        x = self.signature_encode(memory, length)
        latent = self.pred_mlp(x)
        return latent

    def forward(self, memory, length):
        # memory = torch.cat(memory,dim=1)

        x = self.signature_encode(memory, length)
        latent = self.pred_mlp(x)

        pred_joint_logits = []

        if self.single_objective == 2:
            #  optimize the leg and joints
            x = self.relu(self.pred_leg_head_0(latent))
            pred_leg_logit = self.pred_leg_head_1(x)

            for i in range(6):
                x = self.pred_joint_heads[i](latent)
                pred_joint_logits.append(x)

        elif self.single_objective == 3:
            #  optimize the leg and joints
            x = self.relu(self.pred_leg_head_0(latent))
            pred_leg_logit = self.pred_leg_head_1(x)

            for i in range(6):
                x = self.pred_joint_heads_seq[i](latent)
                pred_joint_logits.append(x)

        return pred_leg_logit, pred_joint_logits
