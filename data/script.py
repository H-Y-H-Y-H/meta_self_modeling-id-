import os

import numpy as np
import matplotlib.pyplot as plt


def concat_TASK_data():
    all_data = []
    data_rot = 'robot_f_name/'
    for i in range(30):
        data_i = open(data_rot + 'robot_name_210k_%d.txt'%i).read().strip().split('\n')
        all_data.append(data_i)

    all_data = np.concatenate(all_data)
    print(all_data.shape)
    np.savetxt(data_rot + 'f_robot_names210k.txt',all_data,fmt='%s')

# concat_TASK_data()
def delete_not128265():
    data_root =  '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/robot_sign_data_2/'
    list_di = os.listdir(data_root)
    print(len(list_di))
    # for i in range(len(list128265))

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
    import shutil
    urdf_210k_path = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/robot_urdf_210k/'
    urdf_org_path =  '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/URDF_data/'
    data_root = '/media/ubuntu/yh3187/meta_sign_and_urdf/robot_sign_data_2/'
    urdf_save_root = "/media/ubuntu/yh3187/meta_sign_and_urdf/robot_urdf/"
    urdf_folder = os.listdir(data_root)
    print(len(urdf_folder))

    urdf_210k = os.listdir(urdf_210k_path)
    urdf_org = os.listdir(urdf_org_path)

    for i in range(len(urdf_folder)):
        robot_name = urdf_folder[i]

        if robot_name in urdf_210k:
            urdf_pth = urdf_210k_path + robot_name

        elif robot_name in urdf_org:
            urdf_pth = urdf_org_path + robot_name
        else:
            print('cannot find root',robot_name)
            urdf_pth = None
        print(os.listdir(urdf_pth))
        if robot_name+'.txt' in os.listdir(urdf_pth):
            print("GOT IT")
        else:
            print("NO")
            shutil.rmtree(data_root + robot_name, ignore_errors=False, onerror=None)
            if robot_name in os.listdir(data_root):
                print('still have')
            else:
                print('gone')

            continue
        urdf_root = urdf_pth + '/%s.urdf'%robot_name
        joint_root = urdf_pth + '/%s.txt'%robot_name
        os.makedirs(urdf_save_root+robot_name,exist_ok=True)

        trg_urdf_root = urdf_save_root+robot_name + '/%s.urdf'%robot_name
        trg_joint_root = urdf_save_root+robot_name + '/%s.txt'%robot_name
        shutil.copy(urdf_root,trg_urdf_root)
        shutil.copy(joint_root,trg_joint_root)


# get_urdf()
