import os

import numpy as np
import matplotlib.pyplot as plt


def concat_TASK_data():
    all_data = []
    data_rot = 'robot_f_name/'
    for i in range(30):
        data_i = open(data_rot + 'robot_name_210k_%d.txt' % i).read().strip().split('\n')
        all_data.append(data_i)

    all_data = np.concatenate(all_data)
    print(all_data.shape)
    np.savetxt(data_rot + 'f_robot_names210k.txt', all_data, fmt='%s')


# concat_TASK_data()
def delete_not128265():
    data_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/robot_sign_data_2/'
    list_di = os.listdir(data_root)
    print(len(list_di))
    # for i in range(len(list128265))


def filtered_robotname_list():
    robot_list = open("robot_names35k.txt").read().strip().split('\n')
    done_logger = np.loadtxt('done_logger/done_logger_35k.csv', dtype=int)
    print(np.argsort(done_logger))
    id_rank = np.argsort(done_logger)
    threshold = np.where(done_logger[id_rank] == 0)[0][-1]
    print(done_logger[id_rank[threshold]])

    print(threshold)
    robot_list = np.asarray(robot_list)
    np.savetxt('filtered_%d_from35k.txt' % (threshold + 1), robot_list[id_rank[:(threshold + 1)]], fmt='%s')
    plt.bar(list(np.arange(len(done_logger))), done_logger[id_rank])
    plt.show()


# filtered_robotname_list()


def robot_name_list_generator():
    temp_local_path = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/robot_sign_data_2/'

    file = '16_5_3_2_10_1_5_6_14_11_7_6_18_7_9_10'
    list_di = os.listdir(temp_local_path + file)
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
        txt_file = list_urdf[i] + '.txt'
        sub_folder = os.listdir(urdf_210k_path + list_urdf[i])
        if txt_file not in sub_folder:
            print(sub_folder)
            empty += 1
        else:
            filtered_210k_list.append(list_urdf[i])
    print(len(filtered_210k_list))
    # print(list_urdf)
    np.savetxt('robot_names210k.txt', np.asarray(filtered_210k_list), fmt='%s')


def get_urdf():
    import shutil
    urdf_210k_path = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/robot_urdf_210k/'
    urdf_org_path = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data/URDF_data/'
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
            print('cannot find root', robot_name)
            urdf_pth = None
        print(os.listdir(urdf_pth))
        if robot_name + '.txt' in os.listdir(urdf_pth):
            print("GOT IT")
        else:
            print("NO")
            shutil.rmtree(data_root + robot_name, ignore_errors=False, onerror=None)
            if robot_name in os.listdir(data_root):
                print('still have')
            else:
                print('gone')

            continue
        urdf_root = urdf_pth + '/%s.urdf' % robot_name
        joint_root = urdf_pth + '/%s.txt' % robot_name
        os.makedirs(urdf_save_root + robot_name, exist_ok=True)

        trg_urdf_root = urdf_save_root + robot_name + '/%s.urdf' % robot_name
        trg_joint_root = urdf_save_root + robot_name + '/%s.txt' % robot_name
        shutil.copy(urdf_root, trg_urdf_root)
        shutil.copy(joint_root, trg_joint_root)


# get_urdf()
def unique_leg_conf_idx(robot_names):
    leg_conf_count = dict()
    new_robot_names = []
    for robot_name in robot_names:
        name_code = list(map(int, robot_name.split('_')))
        if name_code[4] < name_code[0]:
            swap_buffer0 = np.copy(name_code[4:8])
            swap_buffer1 = np.copy(name_code[12:16])
            name_code[4:8] = name_code[:4]
            name_code[12:16] = name_code[8:12]

            name_code[:4] = swap_buffer0
            name_code[8:12] = swap_buffer1

        new_name = ''
        for i in range(len(name_code)):
            if i == len(name_code) - 1:
                new_name += str(name_code[i])
            else:
                new_name += str(name_code[i]) + '_'
        new_robot_names.append(new_name)

        leg_code = (name_code[0], name_code[4], name_code[8], name_code[12])

        if leg_code in leg_conf_count:
            leg_conf_count[leg_code] += 1
        else:
            leg_conf_count[leg_code] = 1
    np.savetxt('Apr28_2leg_label_128295.txt', np.asarray(new_robot_names), fmt="%s")
    return leg_conf_count

def get_label():
    num_robot = 108669
    robot_names = open('f_robot_name_%d.txt' % num_robot).read().strip().split('\n')  # [:1000]
    label = unique_leg_conf_idx(robot_names)
    np.savetxt('leg_labels.csv', np.asarray(list(label.keys())), fmt="%i")

    print(len(label), label)

def robot_name_combine():
    done_list = []

    for i in range(21):
        d = np.loadtxt('name_filter/f_robot_name_210k_%d.txt' % i, dtype=str)
        done_list.append(d)

    done_list_all = np.concatenate(done_list)
    print(len(done_list_all))
    np.savetxt("name_filter/May12_robot_name_%d.txt" % len(done_list_all), done_list_all, fmt="%s")

# robot_name_combine()

def filter_robot_pkl_issue():
    # dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data_V2/robot_sign_data_2/'
    dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/data_83726/'
    urdf_folder_list = os.listdir(dataset_root)
    print(len(urdf_folder_list))
    # num_robots = 83726
    # name_list = np.loadtxt('robot_name_rest%d.txt'%num_robots,dtype=str)
    # rest_list = []
    # for i in range(len(urdf_folder_list)):
    #     if urdf_folder_list[i] in name_list:
    #         rest_list.append(urdf_folder_list[i])

    # np.savetxt('robot_name_rest%d.txt'%len(rest_list),np.asarray(rest_list),fmt='%s')
    # check_pth = dataset_root + name_list[i] + '/sans_100_0_V2.npy'

    # try:
    #     arr = np.load(check_pth)
    #     flag = np.isnan(arr)
    #     if flag.any():
    #         print('NaN Detected')
    #         print(i, name_list[i])
    # except:
    #     # error_list.append(name_list[i])
    #     print(i, name_list[i])
    #  10_0_0_11_16_6_10_5_18_6_2_7_14_0_0_1
    #  10_9_3_4_6_1_5_8_9_11_7_4_14_3_9_8
    # break

# filter_robot_pkl_issue()
import shutil
def transfer_urdf_to_target():
    target_urdf_pth = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/robot_urdf/'
    all_old_files = os.listdir(target_urdf_pth)
    urdf_pth = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/robot_urdf_search/'
    all_files = os.listdir(urdf_pth)
    count = 0
    for i in range(len(all_files)):
        ur_name = all_files[i]
        if ur_name in all_old_files:
            count += 1
            print("FK", count)
        else:
            if len(os.listdir(urdf_pth+ur_name)) == 2:
                src1 = urdf_pth + '%s/%s.txt' % (ur_name,  ur_name)
                src2 = urdf_pth + '%s/%s.urdf' % (ur_name, ur_name)

                tgt1 = target_urdf_pth + '%s/%s.txt' % (ur_name, ur_name)
                tgt2 = target_urdf_pth + '%s/%s.urdf' % (ur_name, ur_name)
                os.makedirs(target_urdf_pth+ur_name, exist_ok=True)

                shutil.copyfile(src1, tgt1)
                shutil.copyfile(src2, tgt2)

# transfer_urdf_to_target()

def get_urdf_name_list():
    target_urdf_pth = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/sign_data/'

    # target_urdf_pth = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/robot_urdf/'
    all_old_files = os.listdir(target_urdf_pth)
    # # np.savetxt("Jun3_robot_name_%d.txt" % len(all_old_files), all_old_files, fmt="%s")

    for i in range(len(all_old_files)):
        folder_pth = target_urdf_pth + all_old_files[i]
        files_in_folder = os.listdir(folder_pth)
        if "sans_100_0_V2.npy" not in files_in_folder:
            print('?????????????')
    #         file_loaded = np.load(folder_pth+"/sans_200_0_V2.npy")
    #         print(file_loaded.shape)
    #         file_new = file_loaded[:100]
    #         np.save(folder_pth+'/sans_100_0_V2.npy', file_new)

    # np.savetxt('all_urdf_name_%d.txt' % len(all_old_files), all_old_files, fmt="%s")
    np.savetxt('Jun6_robot_name_%d.txt'%len(all_old_files), all_old_files, fmt="%s")

# get_urdf_name_list()


def transfer_urdf_to_temp_urdf():

    robot_names = np.loadtxt('../meta_sm/test_results/100acc_robo_name.txt', dtype='str')

    urdf_pth = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/robot_urdf/'
    all_files = os.listdir(urdf_pth)
    count = 0

    target_urdf_pth = '../robot_zoo/'
    for i in range(len(robot_names)):
        ur_name = robot_names[i]
        src1 = urdf_pth + '%s/%s.txt'%(ur_name,ur_name)
        src2 = urdf_pth + '%s/%s.urdf'%(ur_name,ur_name)

        tgt1 = target_urdf_pth + '%s/%s.txt'%(ur_name,ur_name)
        tgt2 = target_urdf_pth + '%s/%s.urdf'%(ur_name,ur_name)
        os.makedirs(target_urdf_pth+ur_name,exist_ok=True)

        shutil.copyfile(src1, tgt1)
        shutil.copyfile(src2, tgt2)

transfer_urdf_to_temp_urdf()

def npydata():
    dataset_root = '/home/ubuntu/Desktop/meta_real/data/robot_sign_data/'
    robot_names = os.listdir(dataset_root)
    npy1 = np.load(dataset_root+robot_names[1]+'/sans_100_0_V2.npy')
    print(npy1.shape)

    dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'
    robot_names = open('../data/Jun6_robot_name_200115.txt').read().strip().split('\n')
    robot_names = robot_names[int(0.8 * len(robot_names)):]
    rn = robot_names[0]
    rp = dataset_root + 'sign_data/%s' % rn
    npy2 = np.load(rp + '/sans_100_0_V2.npy')
    print(npy2.shape)

# npydata()

