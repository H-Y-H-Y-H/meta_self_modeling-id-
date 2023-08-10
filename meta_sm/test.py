from train import *

# def get_dataset():

real_test = False
if __name__ == "__main__":
    import wandb
    import argparse
    device = 'cuda:0'
    api = wandb.Api()
    runs = api.runs("robotics/meta_id_dyna")

    # model_name = 'wobbly-sponge-143'
    # model_path = '../data/logger_wobbly-sponge-143/epoch814-acc0.7655'

    # model_name = 'expert-surf-151'
    # model_path = '../data/logger_%s/epoch120-acc0.3172'%model_name

    # model_name = 'lemon-aardvark-152'
    # model_path = '../data/logger_%s/epoch49-acc0.6427'%model_name

    model_name = 'glad-dawn-158'
    model_path = '../data/logger_%s/epoch257-acc0.6214'%model_name

    if real_test:
        dataset_root = '/home/ubuntu/Desktop/meta_real/data/robot_sign_data/'
        robot_names = os.listdir(dataset_root)
    else:
        dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'
        robot_names = open('../data/Jun6_robot_name_200115.txt').read().strip().split('\n')
        robot_names = robot_names[int(0.8*len(robot_names)):]

    idx_sample_flag = 0
    print('idx', idx_sample_flag)
    # robot_names = np.loadtxt('test_results/100acc_robo_name.txt', dtype='str')

    summary_list, config_list, name_list = [], [], []
    config = None
    for run in runs:
        if run.name == model_name:
            print('found: ', model_name)
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}

    config = argparse.Namespace(**config)
    PosEnc = PositionalEncoder(d_input=30, n_freqs=5)

    sign_size = 100
    batch_size = 128
    num_worker = 0
    loss_alpha = 0.25
    d_input = 27

    if real_test:
        robot_paths = dict()
        for rn in robot_names:
            rp = dataset_root + rn
            robot_paths[rn] = rp
    else:
        robot_paths = dict()
        for rn in robot_names:
            rp = dataset_root + 'sign_data/%s' % rn
            robot_paths[rn] = rp


    idx2leg = np.loadtxt('../data/leg_labels.csv')
    leg2idx = dict()
    for i in range(len(idx2leg)):
        leg2idx[tuple(idx2leg[i])] = i

    model = PredConf(state_dim=d_input,
                     MLSTM_hidden_dim=config.MLSTM_hidden_dim,
                     mlp_hidden_dim=config.mlp_hidden_dim,
                     single_objective=config.task,
                     device=device,
                     baseline_id = config.baseline_id
                     )

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()
    # Data you want evaluate

    test_dataset = SASFDataset(robot_paths,
                               robot_names,
                               leg2idx,
                               sign_size=sign_size,
                               max_sample_size=config.max_sample_size,
                               all_sign_flag=config.all_sign_flag,
                               torch_device=device,
                               choose_10steps_input=config.choose_10steps_input,
                               idx_sample_flag = idx_sample_flag,
                               obs_noise=0.0,
                               )

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
    test_b_num = len(robot_names)
    test_joint_acc = 0
    test_leg_acc = 0

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    leg_acc_list, joint_acc_list = [], []
    pred_numpy_list = []
    grth_numpy_list = []
    for batch in tqdm(test_loader):
        memory, gt_leg_cfg, gt_joint_cfg, length = batch
        if config.pos_encoder:
            memory = PosEnc(memory)
        if config.rm_xyz:
            memory = memory[:, :, 3:]
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

            test_result = (pred_joint_cfg_id == gt_joint_cfg).int().view(6, -1).T

            pred_numpy_list.append(pred_joint_cfg_id.detach().cpu().numpy())
            grth_numpy_list.append(gt_joint_cfg.detach().cpu().numpy())

            joint_acc = test_result.sum(dim=1).detach().cpu().numpy()

            leg_acc_list.append(test_correct_leg)
            joint_acc_list.append(joint_acc)

            test_joint_sample_num += len(gt_joint_cfg)
            test_leg_sample_num += len(gt_leg_cfg)

            test_leg_acc += test_correct_leg.sum()
            test_joint_acc += joint_acc.sum()

            test_running_leg_loss /= test_b_num
            test_running_joint_loss /= test_b_num
            test_running_loss /= test_b_num

    log_data = [test_leg_acc / test_b_num,
                test_joint_acc / (test_b_num * 6),
                (test_joint_acc + test_leg_acc) / (test_b_num * 7),
                test_running_leg_loss, test_running_joint_loss, test_running_loss]



    joint_acc_list = np.concatenate(joint_acc_list)
    leg_acc_list = np.concatenate(leg_acc_list)
    pred_numpy_list = np.concatenate(pred_numpy_list)
    grth_numpy_list = np.concatenate(grth_numpy_list)


    result_log_path = 'test_results/model_%s_%d' % (model_name,idx_sample_flag)

    # result_log_path = 'test_real_results/model_%s_%d' % (model_name,idx_sample_flag)
    os.makedirs(result_log_path, exist_ok=True)

    print("results save in: ", result_log_path)
    np.savetxt(result_log_path + '/pred_joint.csv', pred_numpy_list)
    np.savetxt(result_log_path + '/grth_joint.csv', grth_numpy_list)


    np.savetxt(result_log_path + '/acc_leg.csv', np.array(leg_acc_list))
    np.savetxt(result_log_path + '/acc_joint.csv', joint_acc_list)
    np.savetxt(result_log_path + '/test_robot_names.txt', robot_names, fmt='%s')
    np.savetxt(result_log_path + '/logger.csv', np.asarray(log_data))
    print(test_running_leg_loss)
    print(test_running_joint_loss)
    print(test_running_loss)
    print(log_data)
