import numpy as np

dataset_root = '/home/ubuntu/Documents/data_4_meta_self_modeling_id/'
robot_names = open('../../data/Jun6_robot_name_200115.txt').read().strip().split('\n')
robot_names = robot_names[int(0.8 * len(robot_names)):]
model_name = 'model_'+'lemon-aardvark-152'


def acc_robot_name(model_name):
    robo_exist_list = []
    for idx in range(1):
        acc_joint = np.loadtxt(model_name+'_%d/acc_joint.csv'%idx)
        acc_leg = np.loadtxt(model_name+'_%d/acc_leg.csv'%idx)

        leg_right = np.where(acc_leg == 1)[0]
        joint_right = np.where(acc_joint >= 4)[0]

        leg_r_num = len(leg_right)
        joint_r_num = len(joint_right)
        data_num = len(acc_joint)

        j_acc = joint_r_num/data_num
        l_acc = leg_r_num/data_num

        print(j_acc, l_acc)

        passed_robo = np.asarray(joint_right, dtype=np.int32)


        for i in range(data_num):
            if i in joint_right and i in leg_right:
                rn = robot_names[i]
                robo_exist_list.append(rn)

    # filter_list = []
    # for name in robo_exist_list:
    #     if robo_exist_list.count(name) >= 10:
    #         if name not in filter_list:
    #             filter_list.append(name)
    print(len(robo_exist_list)/data_num)
    np.savetxt('100acc_robo_name.txt', robo_exist_list, fmt='%s')

acc_robot_name(model_name)

def joint_pred_eval(model_name):
    model_name=model_name+'_0'
    all_robot = np.loadtxt(model_name+'/test_robot_names.txt',dtype='str')
    pred_joint = np.loadtxt(model_name+'/pred_joint.csv')
    grth_joint = np.loadtxt(model_name+'/grth_joint.csv')
    L1_loss = abs(pred_joint-grth_joint)
    L1_loss = np.where(L1_loss < 6, L1_loss, 12 - L1_loss)
    L1_loss = L1_loss.reshape(6,-1)/6
    test_correct_j = (pred_joint == grth_joint).reshape(6,-1).astype(int)

    mean_l1 = np.mean(L1_loss, axis=1)
    std_l1 = np.std(L1_loss, axis=1)


    mean_rst = np.mean(test_correct_j, axis=1)
    std_rst = np.std(test_correct_j, axis=1)

    print(mean_rst,std_rst)

    np.savetxt(model_name+'/joint_predVSrslt.csv',[mean_rst,std_rst,mean_l1,std_l1])
    print(pred_joint.shape, grth_joint.shape)
    print(len(all_robot))

joint_pred_eval(model_name)

def leg_pred_eval(model_name):
    acc_leg = np.loadtxt(model_name + '/acc_leg.csv' )
    all_robot = np.loadtxt(model_name + '/test_robot_names.txt', dtype='str')
    leg_right = np.where(acc_leg == 1)[0]
    print(leg_right.shape,all_robot.shape)
    print(np.mean(acc_leg),np.std(acc_leg))

# leg_pred_eval(model_name)