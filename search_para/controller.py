import numpy as np
import random
import urdfpy as upy
import pybullet as p
# from env_V3 import *

def random_para():
    para = np.random.uniform(-1,1,size=10)
    para[2:6] *= np.pi

    return para


def batch_random_para(para_batch):
    for i in range(10):
        para_batch[i][i] = random.uniform(-1, 1)
        if i in [2,3,4,5]:
            para_batch[i][i] *= 2*np.pi
    return para_batch

def ik_batch_random_para(para_batch):
    num = len(para_batch)
    max_y_offset = 0.15
    max_z_offset = 0.1
    max_leg_lift_offset = 0.1
    max_small_change = 0.03
    for i in range(1, num): #keep one of the parent
        threshold = random.random()
        if threshold <= 0.2:
            para_batch[i][1] += random.uniform(max(-max_small_change, -para_batch[i][1]), min(max_small_change, max_y_offset-para_batch[i][1]))
        elif threshold <= 0.4:
            para_batch[i][2] += random.uniform(max(-max_small_change, -para_batch[i][2] - 2*max_z_offset), min(max_small_change, max_z_offset-para_batch[i][2]))
        elif threshold <= 0.6:
            para_batch[i][3] += random.uniform(max(-max_small_change, -para_batch[i][3]), min(max_small_change, max_leg_lift_offset-para_batch[i][3]))
        else:
            # 40% of chance changing everything in para
            para_batch[i][1] = random.uniform(0, max_y_offset)
            para_batch[i][2] = random.uniform(-2*max_z_offset, max_z_offset)
            para_batch[i][3] = random.uniform(0, max_leg_lift_offset)

    return para_batch


Period = 16
def sin_move(ti, para):
    assert len(para) == 10, "Action para should be a vector of length 10"
    # print(para)
    s_action = np.zeros(12)
    # print(ti)
    s_action[0] = para[0] * np.sin(ti / Period * 2 * np.pi + para[2]) # right   hind
    s_action[3] = para[1] * np.sin(ti / Period * 2 * np.pi + para[3]) # right  front
    s_action[6] = para[1] * np.sin(ti / Period * 2 * np.pi + para[4]) # left  front
    s_action[9] = para[0] * np.sin(ti / Period * 2 * np.pi + para[5]) # left  hind

    s_action[1] = para[6] * np.sin(ti / Period * 2 * np.pi + para[2]) # right   hind
    s_action[4] = para[7] * np.sin(ti / Period * 2 * np.pi + para[3]) # right   front
    s_action[7] = para[7] * np.sin(ti / Period * 2 * np.pi + para[4]) # left  front
    s_action[10]= para[6] * np.sin(ti / Period * 2 * np.pi + para[5])  # left  hind

    s_action[2] = para[8] * np.sin(ti / Period * 2 * np.pi + para[2]) # right   hind
    s_action[5] = para[9] * np.sin(ti / Period * 2 * np.pi + para[3]) # right   front
    s_action[8] = para[9] * np.sin(ti / Period * 2 * np.pi + para[4]) # left  front
    s_action[11]= para[8] * np.sin(ti / Period * 2 * np.pi + para[5])  # left  hind

    return s_action

'''
Function: getRobotFeetPos, getDeltaTarget, getSingleGaitKeyFrames
          getTriangleGaitKeyFrames, getGaitWorldFeetPos
are used to generate a gait in shape(keyframes, leg_num, 3)
For single gait: keyframes = 2, which are target positions and initial positions
For triangle gait: keyframes = 3 
'''

def getRobotFeetPos(robotEnv):
    cur_feet_states = np.asarray(p.getLinkStates(robotEnv.robotId, robotEnv.feet_index))
    robotEnv.cur_feet_pos = cur_feet_states[:, 4]
    for i in range(robotEnv.num_leg):
        robotEnv.cur_feet_pos[i] -= robotEnv.cur_base_pos
    return robotEnv.cur_feet_pos


'''
Description: Given the learned feet's xyz offsets, get the feet's target local position
'''
def getDeltaTarget(robotEnv, delta_feet_xyz):
    target_delta_feet_pos = []
    for i in range(robotEnv.num_leg):
        target_delta_feet_pos.append(robotEnv.initial_local_feet_pos[i] + delta_feet_xyz[i])
    # print(f"The target delta feet pos is {target_delta_feet_pos}")
    return target_delta_feet_pos


'''
Description: Move between initial feet positions to target feet positions directly
Output: key_frames; shape(key_frame_num, leg_num, xyz)
'''
def getSingleGaitKeyFrames(robotEnv, target_delta_feet_pos):
    # shape = [keyframe_number, 4 (legs), 3 (xyz)]
    gait_key_frames = []
    gait_key_frame1 = np.concatenate(target_delta_feet_pos, axis=None)  # [0,0,...]
    gait_key_frame1 = np.resize(gait_key_frame1, (4, 3))  # [[[0,0,0], [0,0,0], [0,0,0], [0,0,0]]]
    gait_key_frames.append(gait_key_frame1)
    gait_key_frame2 = np.concatenate(robotEnv.initial_local_feet_pos, axis=None)
    gait_key_frame2 = np.resize(gait_key_frame2, (4, 3))
    gait_key_frames.append(gait_key_frame2)
    gait_key_frames = np.asarray(gait_key_frames)
    return gait_key_frames


'''
Description: Move the feet one by one
# O: origin_points; L: lift_points; D: down_points
# Leg 1   2   3   4
# KF  L   O   O   D
# KF  D   L   O   O
# KF  O   D   L   O
# KF  O   O   D   L
Output: key_frames; shape(key_frame_num, leg_num, xyz)
'''
def getTriangleGaitKeyFrames(robotEnv, target_delta_feet_pos, leg_lift):
    origin_points = robotEnv.initial_local_feet_pos
    lift_points = [target_delta_feet_pos[i]+[0, 0, leg_lift] for i in range(len(target_delta_feet_pos))]
    down_points = target_delta_feet_pos
    gait_key_frame1 = [lift_points[0], origin_points[1], origin_points[2], down_points[3]]
    gait_key_frame2 = [down_points[0], lift_points[1], origin_points[2], origin_points[3]]
    gait_key_frame3 = [origin_points[0], down_points[1], lift_points[2], origin_points[3]]
    gait_key_frame4 = [origin_points[0], origin_points[1], down_points[2], lift_points[3]]
    gait_key_frames = []
    gait_key_frames.extend([gait_key_frame1, gait_key_frame2, gait_key_frame3, gait_key_frame4])
    gait_key_frames = np.asarray(gait_key_frames)
    return gait_key_frames

'''
Description: Move the feet symmetrically, feet 1 and 3 move in the same way, feet 2 and 4 move in the same way
# O: origin_points; L: lift_points; D: down_points
Leg 1   2   3   4
KF  L   O   L   O
KF  D   O   D   O
KF  O   L   O   L
KF  O   D   O   D
Output: key_frames; shape(key_frame_num, leg_num, xyz)
'''
def getSymmetricalGaitKeyFrames(robotEnv, target_delta_feet_pos, leg_lift):
    origin_points = robotEnv.initial_local_feet_pos
    lift_points = [target_delta_feet_pos[i]+[0, 0, leg_lift] for i in range(len(target_delta_feet_pos))]
    down_points = target_delta_feet_pos
    gait_key_frame1 = [lift_points[0], origin_points[1], lift_points[2], origin_points[3]]
    gait_key_frame2 = [down_points[0], origin_points[1], down_points[2], origin_points[3]]
    gait_key_frame3 = [origin_points[0], lift_points[1], origin_points[2], lift_points[3]]
    gait_key_frame4 = [origin_points[0], down_points[1], origin_points[2], down_points[3]]
    gait_key_frames = []
    gait_key_frames.extend([gait_key_frame1, gait_key_frame2, gait_key_frame3, gait_key_frame4])
    gait_key_frames = np.asarray(gait_key_frames)
    # print(f"The gait key frames is {gait_key_frames}")
    # print(f"The shape of gait key frames is {np.shape(gait_key_frames)}")
    return gait_key_frames


'''
Description: Given one target key_frame, calculate the target feet world positions
'''
def getGaitWorldFeetPos(robotEnv, gait_key_frames):
    cur_base_pos_rpy = np.concatenate((robotEnv.cur_base_pos, robotEnv.cur_base_ori), axis=None)
    base_HT = upy.xyz_rpy_to_matrix(cur_base_pos_rpy)
    gait_world_feet_pos = []
    for i in range(robotEnv.num_leg):
        target_foot_pos = np.append(gait_key_frames[i, :], 1)
        world_foot_pos = np.dot(base_HT, target_foot_pos)
        gait_world_feet_pos.append(world_foot_pos[:3])
    # print(f"The gait_world_feet_pos is {gait_world_feet_pos}\n")
    return gait_world_feet_pos

'''
Description: Given one target feet world positions, calculate the joints action
'''
def getJointsAction(robotEnv, gait_world_feet_pos):
    joints_action = p.calculateInverseKinematics2(robotEnv.robotId,
                                                  robotEnv.feet_index,
                                                  gait_world_feet_pos,
                                                  solver=robotEnv.ik_solver)

    return joints_action

'''
Description: Get all key_frames for a specific gait, used for IK_move
# input: 
#       feet_offset: [delta_x, delta_y, delta_z, leg_lift]
#       gait: one of the GAIT_TYPE
'''
def getAllKeyFrame(robotEnv, feet_offset, gait):
    expanded_feet_offset = np.asarray([[feet_offset[0], feet_offset[1], feet_offset[2]]] * 2 +
                                      [[-feet_offset[0], feet_offset[1], feet_offset[2]]] * 2)
    leg_lift = feet_offset[3]
    target_delta_feet_pos = getDeltaTarget(robotEnv, expanded_feet_offset)
    if gait == 'single':
        gait_key_frames = getSingleGaitKeyFrames(robotEnv, target_delta_feet_pos)
    elif gait == 'triangle':
        gait_key_frames = getTriangleGaitKeyFrames(robotEnv, target_delta_feet_pos, leg_lift)
    elif gait == 'symmetrical_triangle':
        gait_key_frames = getSymmetricalGaitKeyFrames(robotEnv, target_delta_feet_pos, leg_lift)
    else:
        print("You enter the wrong gait. Exit")
        exit()
    return gait_key_frames

'''
Description: Given one key_frame, calculate the joints action
'''
def IK_move(robotEnv, gait_key_frame):
    gait_world_feet_pos = getGaitWorldFeetPos(robotEnv, gait_key_frame)
    joints_actions = getJointsAction(robotEnv, gait_world_feet_pos)
    joints_actions = np.asarray(joints_actions)
    # print(f"The shape of joints actions is {np.shape(joints_actions)}\n")
    # shape = [key_frame_num, 12]
    return joints_actions

def change_parameters(para):
    for i in range(16):
        rdm_number = random.uniform(-1, 1)
        if random.getrandbits(1):
            if i in [0, 1, 6, 7, 8, 9]:
                para[i] = rdm_number
            elif i in range(2, 6):
                para[i] = 2 * np.pi * rdm_number
            elif i in range(10, 12):
                para[i] = rdm_number * (1 - abs(para[i - 10]))
            elif i in range(12, 16):
                para[i] = rdm_number * (1 - abs(para[i - 6]))
    return para


if __name__ == '__main__':
    para_batch = np.array([random_para()]*16)
    batch_random_para(para_batch)
    print(para_batch)

