import numpy as np

from train import *



# def get_dataset():



if __name__ == "__main__":
    device = 'cuda:0'
    model_name = "logger_128k_256_2"
    model_path = '../data/%s/epoch54-acc0.4815'%model_name
    idx2leg = np.loadtxt('../data/leg_labels0.csv')
    dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'
    robot_names = open('../data/f_robot_names128295.txt').read().strip().split('\n')[:129]
    result_log_path = 'test_results/model_%s'%model_name
    os.makedirs(result_log_path,exist_ok=True)
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
    # idx2leg = list(unique_leg_count.keys())
    # leg2idx = {leg: idx for idx, leg in enumerate(idx2leg)}

    idx2leg = np.loadtxt('../data/leg_labels0.csv')
    leg2idx = dict()
    for i in range(len(idx2leg)):
        leg2idx[tuple(idx2leg[i])] = i

    print("Num of test unique conf:", len(unique_leg_count))


    shuffle(robot_names)

    # split_idx = int(len(robot_names) * 0.8)
    split_idx = 0
    # train_robot_names = robot_names[:split_idx]
    test_robot_names = robot_names[split_idx:]
    print(len(test_robot_names), "robots will be tested")
    # train_unique_leg_count = unique_leg_conf_idx(train_robot_names)
    test_unique_leg_count = unique_leg_conf_idx(test_robot_names)


    model = PredConf(state_dim=28,
                     MLSTM_hidden_dim=256,
                     mlp_hidden_dim=256,
                     encoder_type=0,
                     single_objective=2,
                     device=device
                     )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    model.eval()
    model = model.cuda()

    # Data you want evaluate
    # robot_names = get_dataset(robot_names)
    test_dataset = SASFDataset(robot_paths, test_robot_names, leg2idx, sign_size=sign_size,
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
    test_joint_acc = 0
    test_leg_acc = 0

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    leg_acc_list, joint_acc_list = [], []
    for i, batch in enumerate(test_loader):
        memory, gt_leg_cfg, gt_joint_cfg, length = batch
        current_N = len(memory)
        print('test_robot_name:',test_robot_names[current_N*i:current_N*(i+1)])

        memory = memory.to(device)
        gt_leg_cfg = gt_leg_cfg.to(device)
        gt_joint_cfg = gt_joint_cfg.to(device)

        with torch.no_grad():
            pred_leg_cfg, pred_joint_cfg = model(memory, length) # N x 30;  6 x N x 12

            leg_loss = criterion1(pred_leg_cfg, gt_leg_cfg)
            pred_joint_cfg = torch.cat(pred_joint_cfg)
            print(pred_joint_cfg.shape)
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


            test_result = (pred_joint_cfg_id == gt_joint_cfg).int().view(-1,6)
            test_result = test_result.sum(dim=1).detach().cpu().numpy()
            leg_acc_list.append(test_correct_leg)
            joint_acc_list.append(test_result)

            test_joint_sample_num += len(gt_joint_cfg)
            test_leg_sample_num += len(gt_leg_cfg)
        test_b_num += current_N

        test_leg_acc += test_correct_leg.sum()
        test_joint_acc += test_result.sum()

        test_running_leg_loss   /= test_b_num
        test_running_joint_loss /= test_b_num
        test_running_loss       /= test_b_num

    log_data = [test_leg_acc/test_b_num,
                test_joint_acc/(test_b_num*6),
                (test_joint_acc+test_leg_acc)/(test_b_num*7),
                test_running_leg_loss,test_running_joint_loss,test_running_loss]

    joint_acc_list = np.concatenate(joint_acc_list)
    leg_acc_list = np.concatenate(leg_acc_list)
    np.savetxt(result_log_path+'/acc_leg.csv', np.array(leg_acc_list))
    np.savetxt(result_log_path+'/acc_joint.csv', joint_acc_list)
    np.savetxt(result_log_path+'/test_robot_names.txt', test_robot_names,fmt='%s')
    np.savetxt(result_log_path+'/logger.csv', np.asarray(log_data))
    print(test_running_leg_loss)
    print(test_running_joint_loss)
    print(test_running_loss)
    print(log_data)




