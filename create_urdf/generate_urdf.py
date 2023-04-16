import os

import pybullet as p
import time
import pybullet_data
import numpy as np
import creater
import shutil
import diff_function as dfun
from tqdm import tqdm
import time


def create_initial_q():
    initial_q1 = np.random.random_sample(size=5) * 2 - 1
    initial_q2 = np.random.random_sample(size=5) * 2 - 1
    initial_q1 *= np.pi / 3
    initial_q2 *= np.pi / 3
    initial_q = np.hstack((initial_q1, initial_q2, initial_q2, initial_q1))
    initial_q[[0, 4, 5, 9, 10, 14, 15, 19]] = 0
    return initial_q


def main():
    existing_urdf1 = set(os.listdir('../data/robot_urdf_10k'))
    existing_urdf2 = set(os.listdir('../data/robot_urdf_40k'))
    existing_urdf3 = set(os.listdir('../data/robot_urdf_210k'))
    existing_urdf = existing_urdf1 | existing_urdf2 | existing_urdf3

    SYMMETRY = True
    GUI4DEBUG = False
    NUM_ROBOT = 20000
    find_count = 0
    fail_count = 0
    duplicate_count = 0

    NUM_INITIAL_ANGLE = 30

    data_folder = '../data'
    urdf_store_folder = 'robot_urdf_search'

    os.makedirs(os.path.join(data_folder, urdf_store_folder), exist_ok=True)

    # Start simulation
    if GUI4DEBUG:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)

    for i in range(NUM_ROBOT):
        # while True:
        if SYMMETRY:
            filename = creater.create_symm_notop()
        else:
            filename = creater.create()

        if filename in existing_urdf or filename is None:
            print(f"{filename} Existed")
            duplicate_count += 1
            continue

        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")

        urdf_folder = os.path.join(data_folder, urdf_store_folder, filename)
        save_pth = os.path.join(urdf_folder, filename)
        savearray = []

        for n in range(NUM_INITIAL_ANGLE):
            secondtest = False
            startPos = [0, 0, 1]
            startOrientation = p.getQuaternionFromEuler([0, 0, 0])
            roboID = p.loadURDF(urdf_folder + "/%s.urdf" % filename, startPos,
                                startOrientation, useFixedBase=1,
                                flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

            initial_q = create_initial_q()

            checkflag = True
            checkflag_ik = False

            for j in range(20):
                p.setJointMotorControl2(roboID, j, p.POSITION_CONTROL,
                                        targetPosition=initial_q[j], force=2, maxVelocity=100)

            # set the center of mass frame (loadURDF sets base link frame) s
            # tartPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
            for m in range(60):
                p.stepSimulation()

            # Check self-collision:
            checkarray = []
            for k in range(20):
                if k in [0, 4, 5, 9, 10, 14, 15, 19]:
                    checkarray.append(0)
                else:
                    b = p.getJointState(roboID, k)
                    b = b[0]
                    checkarray.append(b)
                    if b - initial_q[k] > 0.05:

                        # DEBUG: If you turn on the GUI, you can see the leg movements and if there is self-collision,
                        # the program will stop and you have 3 seconds to see if it really happened.
                        if GUI4DEBUG == True:
                            for i in range(30):
                                p.stepSimulation()
                                time.sleep(1 / 240)

                        # print("SELF-COLLISION!")
                        checkflag = False

            # Check feet position
            ###############
            leg_info = []
            for ii in range(4, 20, 5):
                link = p.getLinkState(roboID, ii)
                tran_world = link[0]
                leg_info.append(link[0])
                euler = p.getEulerFromQuaternion(link[1])
                vect = dfun.get_uv(euler)
                z_offset = startPos[2] - (102.7 / 1000)

                leg_x = tran_world[0]
                leg_y = tran_world[1]
                leg_z = tran_world[2]

                if (leg_z) > z_offset:
                    # print(ii, "z error")
                    # time.sleep(3)
                    checkflag = False

                if vect[2] <= 0.05:
                    # print(ii, "uv error")
                    # time.sleep(3)
                    checkflag = False
            ###############

            x_1 = leg_info[0][0]
            y_1 = leg_info[0][1]
            z_1 = leg_info[0][2]

            x_2 = leg_info[1][0]
            y_2 = leg_info[1][1]
            z_2 = leg_info[1][2]

            x_3 = leg_info[2][0]
            y_3 = leg_info[2][1]
            z_3 = leg_info[2][2]

            x_4 = leg_info[3][0]
            y_4 = leg_info[3][1]
            z_4 = leg_info[3][2]

            initial_q = initial_q.tolist()
            base_info = p.getBasePositionAndOrientation(roboID)

            q_final = []
            if checkflag == True:
                for mm in range(10):
                    if checkflag_ik == True:
                        break
                    for nn in range(10):
                        if checkflag_ik == True:
                            break
                        checkflag = True
                        step = 0.01
                        gap = step * 10 / 2
                        ik_list1 = p.calculateInverseKinematics(
                            roboID, 4, [(x_1 - gap + mm * step), (y_1 - gap + nn * step), startPos[2] - 0.15])
                        ik_list2 = p.calculateInverseKinematics(
                            roboID, 9, [(x_2 - gap + mm * step), (y_2 - gap + nn * step), startPos[2] - 0.15])

                        q1 = [float(0), ik_list1[0], ik_list1[1],
                              ik_list1[2], float(0)]
                        q2 = [float(0), ik_list2[3], ik_list2[4],
                              ik_list2[5], float(0)]
                        q1_q2 = q1 + q2
                        q_p = q2 + q1
                        q_final = q1_q2 + q_p
                        # print(q_final)
                        for j in range(len(q_final)):
                            p.setJointMotorControl2(roboID, j, p.POSITION_CONTROL,
                                                    targetPosition=q_final[j], force=2, maxVelocity=100)

                        for k in range(60):
                            p.stepSimulation()

                        check_list1 = p.getLinkState(roboID, 4)
                        check_list2 = p.getLinkState(roboID, 9)
                        check_list3 = p.getLinkState(roboID, 14)
                        check_list4 = p.getLinkState(roboID, 19)
                        base_info = p.getBasePositionAndOrientation(roboID)

                        # print(check_list)
                        z_check1 = check_list1[0][2]
                        z_check2 = check_list2[0][2]
                        z_check = max(
                            abs(z_check1 - (startPos[2] - 0.15)), abs(z_check2 - (startPos[2] - 0.15)))
                        if abs(z_check) < 0.03:
                            # set new value and save?
                            # print("GET1")
                            # np.savetxt(save_pth + "2.txt", np.asarray(q_final))
                            checkflag_ik = True
                            # print(base_info[0], check_list1[0],
                            #      check_list2[0], check_list3[0], check_list4[0])
                            # tri = dfun.tri_leg_balance(
                            #    [0, 0, 0], check_list1[0], check_list2[0], check_list3[0], check_list4[0])
                            initial_q = q_final
                            # if tri is False:
                            #    # print("gg")
                            #    checkflag = False
                            # initial_q = initial_q.tolist()
                            for h in range(4, 20, 5):
                                # print("gg2")
                                ppp = p.getLinkState(roboID, h)
                                euler2 = p.getEulerFromQuaternion(ppp[1])
                                uv = dfun.get_uv(euler2)
                                if uv[2] <= 0.05:
                                    checkflag = False
                                # else:
                                # print("SOLUTION!")

            if checkflag and checkflag_ik:
                secondtest = True

            if secondtest:
                p.resetSimulation()
                p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
                p.setGravity(0, 0, -10)
                planeId = p.loadURDF("plane.urdf")

                roboID2 = p.loadURDF(urdf_folder + "/%s.urdf" % filename, [0, 0, 0.25],
                                     startOrientation, useFixedBase=0,
                                     flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

                for j in range(20):
                    p.setJointMotorControl2(roboID2, j, p.POSITION_CONTROL,
                                            targetPosition=initial_q[j], force=2, maxVelocity=100)
                # set the center of mass frame (loadURDF sets base link frame)
                # startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)

                ori = p.getBasePositionAndOrientation(roboID2)
                ori = ori[1]
                ori = p.getEulerFromQuaternion(ori)

                pos_check_old = np.asarray(p.getBasePositionAndOrientation(roboID2)[0])
                for k in range(1000):
                    p.stepSimulation()
                    pos_check = np.asarray(p.getBasePositionAndOrientation(roboID2)[0])
                    if (k + 1) % 50 == 0:
                        move_value = np.sum((pos_check - pos_check_old) ** 2)
                        if move_value < 0.00001:
                            break
                        else:
                            pos_check_old = pos_check

                ori2 = p.getBasePositionAndOrientation(roboID2)
                ori2 = ori2[1]
                pos_b = p.getBasePositionAndOrientation(roboID2)[0]
                ori2 = p.getEulerFromQuaternion(ori2)

                check_list1 = p.getLinkState(roboID2, 4)
                check_list2 = p.getLinkState(roboID2, 9)
                check_list3 = p.getLinkState(roboID2, 14)
                check_list4 = p.getLinkState(roboID2, 19)

                z_1 = check_list1[0][2]
                z_2 = check_list2[0][2]
                z_3 = check_list3[0][2]
                z_4 = check_list4[0][2]

                z_offset = (pos_b[2] - (102.7 / 1000) - 0.03)
                flag = True

                for nn in range(20):
                    joint = p.getJointState(roboID2, nn)[0]
                    if joint - initial_q[nn] > 0.05:
                        flag = False

                if (abs(ori2[0]) < 0.4) and (abs(ori2[1]) < 0.4) and (abs(ori2[2]) < 0.4):
                    if (z_1 < z_offset) and (z_2 < z_offset) and (flag is True):
                        savearray.append(initial_q)
                        break

            p.resetSimulation()

        if len(savearray) >= 1:
            np.savetxt(save_pth + ".txt", np.asarray(savearray))
            find_count += 1
            print(
                f"Find Config {filename}, find count {find_count}, fail count {fail_count}, duplicate count {duplicate_count}")
        else:
            shutil.rmtree(urdf_folder)
            fail_count += 1

    p.disconnect()


if __name__ == '__main__':
    main()
