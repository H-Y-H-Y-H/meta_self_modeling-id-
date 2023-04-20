import os

import numpy as np
import matplotlib.pyplot as plt


def concat_TASK_data():
    all_data = []
    data_rot = 'robot_f_name/'
    for i in range(35):
        data_i = open(data_rot + 'robot_name_list_%d.txt'%i).read().strip().split('\n')
        all_data.append(data_i)

    all_data = np.concatenate(all_data)
    print(all_data.shape)
    np.savetxt(data_rot + 'f_robot_names35k.txt',all_data,fmt='%s')

# concat_TASK_data()

def filtered_robotname_list():
    robot_list = open("robot_names35k.txt").read().strip().split('\n')
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
# filtered_robotname_list()


def robot_name_list_generator():
    temp_local_path = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/robot_sign_data_2/'

    file = '16_5_3_2_10_1_5_6_14_11_7_6_18_7_9_10'
    list_di = os.listdir(temp_local_path+file)
    print(list_di)
    # list_all_config = os.listdir(temp_local_path+'data/URDF_data/')[35000:]
    # # np.savetxt('robot_names35kto260k.txt',np.asarray(list_all_config),fmt='%s')
    # print(len(list_all_config))
# robot_name_list_generator()

def fileterd_210k_list():
    urdf_210k_path = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/robot_urdf_210k/'
    list_urdf = os.listdir(urdf_210k_path)
    empty = 0

    filtered_210k_list = []
    for i in range(len(list_urdf)):
        txt_file =list_urdf[i] + '.txt'
        sub_folder = os.listdir(urdf_210k_path+list_urdf[i])
        if txt_file not in sub_folder:
            print(sub_folder)
            empty += 1
        else:
            filtered_210k_list.append(list_urdf[i])
    print(len(filtered_210k_list))
    # print(list_urdf)
    np.savetxt('robot_names210k.txt',np.asarray(filtered_210k_list),fmt='%s')

def get_urdf():
    data_root = '/media/ubuntu/yh3187'
    urdf_folder = os.listdir(data_root)
    print(len(urdf_folder))


