import numpy as np

dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'
robot_names = open('../../data/Jun6_all_urdf_name_163648.txt').read().strip().split('\n')
robot_names = robot_names[int(0.8 * len(robot_names)):]

model_name = 'model_upbeat-valley-130'
acc_joint = np.loadtxt(model_name+'/acc_joint.csv')
acc_leg = np.loadtxt(model_name+'/acc_leg.csv')

leg_right = np.where(acc_leg == 1)[0]
joint_right = np.where(acc_joint >= 6)[0]

leg_r_num = len(leg_right)
joint_r_num = len(joint_right)
data_num = len(acc_joint)

j_acc = joint_r_num/data_num
l_acc = leg_r_num/data_num

print(j_acc, l_acc)

passed_robo = np.asarray(joint_right, dtype=np.int32)

robo_exist_list = []
for i in range(data_num):
    if i in joint_right and i in leg_right:
        rn = robot_names[i]
        robo_exist_list.append(rn)

np.savetxt('100acc_robo_name.txt', robo_exist_list, fmt='%s')

