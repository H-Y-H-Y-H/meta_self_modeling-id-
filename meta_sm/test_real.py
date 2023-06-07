from train import *
import wandb
import argparse


def run_model():
    robo_name = "10_9_9_6_11_9_9_6_13_3_3_6_14_3_3_6"
    data_path = '/home/ubuntu/Desktop/meta_real/data/robot_sign_data/%s/'%robo_name
    sign_data = np.load(data_path + 'sans_100_0_V2.npy')


    api = wandb.Api()
    runs = api.runs("robotics/meta_id_dyna")

    model_name = 'hearty-energy-132'
    device = 'cuda:0'
    model_path = '../data/logger_%s/epoch28-acc0.5938' % model_name

    idx_sample_flag = 0
    
    summary_list, config_list, name_list = [], [], []
    config = None
    for run in runs:
        if run.name == model_name:
            print('found: ', model_name)
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}

    config = argparse.Namespace(**config)

    idx2leg = np.loadtxt('../data/leg_labels.csv')
    leg2idx = dict()
    for i in range(len(idx2leg)):
        leg2idx[tuple(idx2leg[i])] = i

    robot_paths = dict()


    model = PredConf(state_dim=30,
                     MLSTM_hidden_dim=config.MLSTM_hidden_dim,
                     mlp_hidden_dim=config.mlp_hidden_dim,
                     single_objective=config.task,
                     device=device)

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    test_dataset = SASFDataset(robot_paths,
                               robot_names,
                               leg2idx,
                               sign_size=sign_size,
                               max_sample_size=config.max_sample_size,
                               all_sign_flag=config.all_sign_flag,
                               torch_device=device,
                               choose_10steps_input=config.choose_10steps_input,
                               idx_sample_flag = idx_sample_flag,
                               obs_noise=0.01)



if __name__ == '__main__':
    run_model()