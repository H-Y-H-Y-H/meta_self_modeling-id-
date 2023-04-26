from train import *



# def get_dataset():



if __name__ == "__main__":
    device = 'cuda:0'
    model_path = '../data/logger_128k/epoch566-acc0.3402'
    idx2leg = np.loadtxt('../data/leg_labels0.csv')
    dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'
    robot_names = open('../data/f_robot_names128295.txt').read().strip().split('\n')#[:1000]
    max_sample_size = 201
    sign_size = 201
    batch_size = 128
    num_worker = 5
    loss_alpha = 0.25


    robot_paths = dict()
    for rn in robot_names:
        rp = dataset_root + 'data/robot_sign_data_2/%s' % rn
        robot_paths[rn] = rp
    unique_leg_count = unique_leg_conf_idx(robot_names)
    idx2leg = list(unique_leg_count.keys())
    leg2idx = {leg: idx for idx, leg in enumerate(idx2leg)}

    shuffle(robot_names)

    # split_idx = int(len(robot_names) * 0.8)
    split_idx = 0
    # train_robot_names = robot_names[:split_idx]
    test_robot_names = robot_names[split_idx:]
    # train_unique_leg_count = unique_leg_conf_idx(train_robot_names)
    test_unique_leg_count = unique_leg_conf_idx(test_robot_names)


    model = PredConf(state_dim=28,
                     MLSTM_hidden_dim=256,
                     mlp_hidden_dim=256,
                     encoder_type=3,
                     single_objective=1,
                     device=device
                     )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()
    model = model.cuda()

    # Data you want evaluate
    # robot_names = get_dataset(robot_names)
    test_dataset = SASFDataset(robot_paths, test_unique_leg_count, leg2idx, sign_size=sign_size,
                                max_sample_size=max_sample_size, all_sign_flag=True,torch_device=device)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_worker)  # , collate_fn=train_dataset.collate

    test_running_loss = 0.0
    test_running_joint_loss = 0.0
    test_running_leg_loss = 0.0

    test_correct_leg = 0.0
    test_correct_joint = 0.0
    test_joint_sample_num = 0.0
    test_leg_sample_num = 0.0
    test_b_num = 0

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()

    for batch in tqdm(test_loader, desc="Testing"):
        memory, gt_leg_cfg, gt_joint_cfg, length = batch
        memory = memory.to(device)
        gt_leg_cfg = gt_leg_cfg.to(device)
        gt_joint_cfg = gt_joint_cfg.to(device)

        with torch.no_grad():
            pred_leg_cfg, pred_joint_cfg = model(memory, length)

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
            test_correct_leg += (pred_leg_cfg_id == gt_leg_cfg).sum().item()
            test_correct_joint += (pred_joint_cfg_id == gt_joint_cfg).sum().item()
            test_joint_sample_num += len(gt_joint_cfg)
            test_leg_sample_num += len(gt_leg_cfg)
        test_b_num+=1

        test_acc = (test_correct_joint+test_correct_leg)/(test_joint_sample_num+test_leg_sample_num)
        test_joint_acc = test_correct_joint/test_joint_sample_num
        test_leg_acc = test_correct_leg/test_leg_sample_num
        test_running_leg_loss   /= test_b_num
        test_running_joint_loss /= test_b_num
        test_running_loss       /= test_b_num





