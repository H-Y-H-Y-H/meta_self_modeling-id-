import os

import numpy as np
import matplotlib.pyplot as plt


def concat_TASK_data():
    all_data = []
    data_rot = 'done_logger/'
    for i in range(35):
        data_i = np.loadtxt(data_rot + 'done_logger35kto260k_%d.csv'%i)
        all_data.append(data_i)

    all_data = np.concatenate(all_data)
    print(all_data.shape)
    np.savetxt(data_rot + 'done_logger_35k.csv',all_data,fmt='%i')

concat_TASK_data()

def filtered_robotname_list():
    robot_list = open("../data/robot_names35k.txt").read().strip().split('\n')
    done_logger = np.loadtxt('done_logger/done_logger_35k.csv', dtype=int)
    print(np.argsort(done_logger))
    id_rank = np.argsort(done_logger)
    threshold = np.where(done_logger[id_rank]==0)[0][-1]
    print(done_logger[id_rank[threshold]])

    print(threshold)
    robot_list = np.asarray(robot_list)
    np.savetxt('filtered_%d_from35k.txt'%(threshold+1),robot_list[id_rank[:(threshold+1)]],fmt='%s')
    plt.bar(list(np.arange(len(done_logger))),done_logger[id_rank])
    plt.show()
filtered_robotname_list()


def robot_name_list_generator():
    temp_local_path = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'

    list_all_config = os.listdir(temp_local_path+'data/URDF_data/')[35000:]
    # np.savetxt('robot_names35kto260k.txt',np.asarray(list_all_config),fmt='%s')
    print(len(list_all_config))
# robot_name_list_generator()