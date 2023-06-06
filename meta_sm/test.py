import numpy as np

from train import *

# def get_dataset():


if __name__ == "__main__":
    import wandb
    import argparse

    api = wandb.Api()
    runs = api.runs("robotics/meta_id_dyna")

    model_name = 'upbeat-valley-130'
    device = 'cuda:0'
    model_path = '../data/logger_%s/epoch2-acc0.5611' % model_name
    dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'

    robot_names = open('../data/Jun6_all_urdf_name_163648.txt').read().strip().split('\n')
    robot_names = robot_names[int(0.8*len(robot_names)):]

    # robot_names = np.loadtxt('test_results/100acc_robo_name.txt', dtype='str')[:10]

    result_log_path = 'test_results/model_%s' % model_name
    os.makedirs(result_log_path, exist_ok=True)

    summary_list, config_list, name_list = [], [], []
    config = None
    for run in runs:
        if run.name == model_name:
            print('found: ', model_name)
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}

    config = argparse.Namespace(**config)

    sign_size = 100
    batch_size = 128
    num_worker = 0
    loss_alpha = 0.25

    robot_paths = dict()
    for rn in robot_names:
        rp = dataset_root + 'sign_data/%s' % rn
        robot_paths[rn] = rp
    unique_leg_count = unique_leg_conf_idx(robot_names)
    # idx2leg = list(unique_leg_count.keys())
    # leg2idx = {leg: idx for idx, leg in enumerate(idx2leg)}

    idx2leg = np.loadtxt('../data/leg_labels.csv')
    leg2idx = dict()
    for i in range(len(idx2leg)):
        leg2idx[tuple(idx2leg[i])] = i

    print("Num of test unique conf:", len(unique_leg_count))

    # split_idx = int(len(robot_names) * 0.8)
    split_idx = 0
    # train_robot_names = robot_names[:split_idx]
    test_robot_names = robot_names[split_idx:]
    print(len(test_robot_names), "robots will be tested")
    # train_unique_leg_count = unique_leg_conf_idx(train_robot_names)
    test_unique_leg_count = unique_leg_conf_idx(test_robot_names)

    model = PredConf(state_dim=30,
                     MLSTM_hidden_dim=config.MLSTM_hidden_dim,
                     mlp_hidden_dim=config.mlp_hidden_dim,
                     single_objective=config.task,
                     device=device)

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()
    # Data you want evaluate

    test_dataset = SASFDataset(robot_paths,
                               test_robot_names,
                               leg2idx,
                               sign_size=sign_size,
                               max_sample_size=config.max_sample_size,
                               all_sign_flag=config.all_sign_flag,
                               torch_device=device,
                               choose_10steps_input=config.choose_10steps_input,
                               obs_noise=0.0)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_worker)  # , collate_fn=train_dataset.collate

    test_running_loss = 0.0
    test_running_joint_loss = 0.0
    test_running_leg_loss = 0.0

    test_correct_leg = 0.0
    test_correct_joint = 0.0
    test_joint_sample_num = 0.0
    test_leg_sample_num = 0.0
    test_b_num = len(test_robot_names)
    test_joint_acc = 0
    test_leg_acc = 0

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    leg_acc_list, joint_acc_list = [], []
    for batch in tqdm(test_loader):
        memory, gt_leg_cfg, gt_joint_cfg, length = batch
        memory = memory.to(device)
        gt_leg_cfg = gt_leg_cfg.to(device)
        gt_joint_cfg = gt_joint_cfg.to(device)

        with torch.no_grad():

            # memory: batch x length x channels
            pred_leg_cfg, pred_joint_cfg = model(memory, length)  # N x 30;

            leg_loss = criterion1(pred_leg_cfg, gt_leg_cfg)
            pred_joint_cfg = torch.cat(pred_joint_cfg)
            gt_joint_cfg = gt_joint_cfg.T.flatten()

            joint_loss = criterion2(pred_joint_cfg, gt_joint_cfg)
            loss = (loss_alpha * leg_loss + (1 - loss_alpha) * joint_loss)

            test_running_leg_loss += leg_loss.item()
            test_running_joint_loss += joint_loss.item()
            test_running_loss += loss.item()

            pred_leg_cfg_id = torch.argmax(pred_leg_cfg, dim=1)
            pred_joint_cfg_id = torch.argmax(pred_joint_cfg, dim=1)

            test_correct_leg = (pred_leg_cfg_id == gt_leg_cfg).int()
            test_correct_leg = test_correct_leg.detach().cpu().numpy()

            test_result = (pred_joint_cfg_id == gt_joint_cfg).int().view(6,-1).T # 16 x 6
            test_result = test_result.sum(dim=1).detach().cpu().numpy() # 16 x 1

            leg_acc_list.append(test_correct_leg)
            joint_acc_list.append(test_result)

            test_joint_sample_num += len(gt_joint_cfg)
            test_leg_sample_num += len(gt_leg_cfg)

        test_leg_acc += test_correct_leg.sum()
        test_joint_acc += test_result.sum()

        test_running_leg_loss /= test_b_num
        test_running_joint_loss /= test_b_num
        test_running_loss /= test_b_num

    log_data = [test_leg_acc / test_b_num,
                test_joint_acc / (test_b_num * 6),
                (test_joint_acc + test_leg_acc) / (test_b_num * 7),
                test_running_leg_loss, test_running_joint_loss, test_running_loss]

    joint_acc_list = np.concatenate(joint_acc_list)
    leg_acc_list = np.concatenate(leg_acc_list)
    print("results save in: ", result_log_path)
    np.savetxt(result_log_path + '/acc_leg.csv', np.array(leg_acc_list))
    np.savetxt(result_log_path + '/acc_joint.csv', joint_acc_list)
    np.savetxt(result_log_path + '/test_robot_names.txt', test_robot_names, fmt='%s')
    np.savetxt(result_log_path + '/logger.csv', np.asarray(log_data))
    print(test_running_leg_loss)
    print(test_running_joint_loss)
    print(test_running_loss)
    print(log_data)
