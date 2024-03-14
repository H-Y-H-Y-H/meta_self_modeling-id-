import numpy as np
import os
import shutil

def transfer_urdf_to_temp_urdf():
    # robot_names = np.loadtxt('../meta_sm/test_results/100acc_robo_name.txt', dtype='str')

    fully_acc_name_list = open('../meta_sm/test_results/100acc_robo_name.txt').read().strip().split('\n')
    print(fully_acc_name_list[0])


    urdf_pth = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/robot_urdf/'
    target_urdf_pth = '100acc_urdf/'

    count = 0

    # target_urdf_pth = '../robot_zoo/'
    for i in range(100):
        ur_name = fully_acc_name_list[i]
        src1 = urdf_pth + '%s/%s.txt'%(ur_name,ur_name)
        src2 = urdf_pth + '%s/%s.urdf'%(ur_name,ur_name)

        tgt1 = target_urdf_pth + '%s/%s.txt'%(ur_name,ur_name)
        tgt2 = target_urdf_pth + '%s/%s.urdf'%(ur_name,ur_name)
        os.makedirs(target_urdf_pth+ur_name, exist_ok=True)

        shutil.copyfile(src1, tgt1)
        shutil.copyfile(src2, tgt2)




transfer_urdf_to_temp_urdf()