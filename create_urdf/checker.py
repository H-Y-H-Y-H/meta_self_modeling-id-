import pybullet as p
import time
import pybullet_data
import numpy as np
import creater
import shutil
import diff_function as dfun
from cal_single import *

def check_urdf_can_stand(filename, initial_q, GUI4DEBUG=False, DELETE_IF_FAIL=True):
    """
    filename (str): conf filename, example 6_0_0_3_11_4_11_10_13_8_1_2_9_0_0_9
    initial_q (np.array): initial joint angle of shape (20,)
    """
    physicsClient = p.connect(p.GUI if GUI4DEBUG else p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")

    urdf_folder = "../data/robot_warehouse/" + filename
    save_pth = urdf_folder + "/" + filename

    savearray = []
    secondtest = False
    # Loading robots
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    roboID = p.loadURDF(urdf_folder + "/%s.urdf" % filename, startPos,
                        startOrientation, useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION or p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT)

    for j in range(20):
        p.setJointMotorControl2(roboID, j, p.POSITION_CONTROL,
                                targetPosition=initial_q[j], force=2, maxVelocity=100)
    # set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    for m in range(60):
        p.stepSimulation()

    # Check self-collision:
    checkarray = []
    checkflag = True
    checkflag_ik = False

    for k in range(20):
        if(k == 0 or k == 4 or k == 5 or k == 9 or k == 10 or k == 14 or k == 15 or k == 19):
            checkarray.append(float(0))
            # print('fine')
        else:
            b = p.getJointState(roboID, k)
            b = b[0]
            checkarray.append(b)
            #print(b, initial_q[k])
            if b - initial_q[k] > 0.05:

                # DEBUG: If you turn on the GUI, you can see the leg movements and if there is self-collision,
                # the program will stop and you have 3 seconds to see if it really happened.
                if GUI4DEBUG == True:
                    for i in range(30):
                        p.stepSimulation()
                        time.sleep(1. / 240.)

                # print("SELF-COLLISION!")
                checkflag = False

    # Check feet position
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
            checkflag = False

        if vect[2] <= 0.05:
            checkflag = False

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
                z_check1 = check_list1[0][2]
                z_check2 = check_list2[0][2]
                z_check = max(
                    abs(z_check1 - (startPos[2] - 0.15)), abs(z_check2 - (startPos[2] - 0.15)))
                if abs(z_check) < 0.03:
                    checkflag_ik = True
                    initial_q = q_final
                    for h in range(4, 20, 5):
                        ppp = p.getLinkState(roboID, h)
                        euler2 = p.getEulerFromQuaternion(ppp[1])
                        uv = dfun.get_uv(euler2)
                        if uv[2] <= 0.05:
                            checkflag = False

    if (checkflag is True) and (checkflag_ik is True):
        secondtest = True

    if secondtest == True:
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
        ori = p.getBasePositionAndOrientation(roboID2)
        ori = ori[1]
        ori = p.getEulerFromQuaternion(ori)
        pos_check_old = np.asarray(p.getBasePositionAndOrientation(roboID2)[0])
        for k in range(1000):
            p.stepSimulation()
            pos_check = np.asarray(p.getBasePositionAndOrientation(roboID2)[0])
            if ((k + 1) % 50 == 0):
                move_value = np.sum((pos_check - pos_check_old) ** 2)
                if (move_value < 0.00001):
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

    p.resetSimulation()
    if len(savearray) >= 1:
        np.savetxt(save_pth + ".txt", np.asarray(savearray))
        print("Success", filename)
    elif DELETE_IF_FAIL:
        print("Failed", filename)
        shutil.rmtree(urdf_folder)

    p.disconnect()

if __name__ == '__main__':
    # test_conf_code = [9, 4, 2, 4, 10, 6, 9, 7, 13, 5, 2, 4, 16, 6, 10, 6]

    filename = '11_10_5_0_10_8_3_2_14_4_9_10_13_2_7_0'
    test_conf_code = list(map(int, filename.split('_')))
    F1 = test_conf_code[:4]
    F2 = test_conf_code[4:8]
    F3 = test_conf_code[8:12]
    F4 = test_conf_code[12:]

    # initial_q = np.zeros((4,5))
    # joint_angle = [0.012283206, 0.040059432, -0.017774746, -0.06222333, 0.055202153, -0.012573212, -0.07586604, 0.056019664, -0.01802823, -0.0014711916, 0.03397146, -0.0065170415]
    # joint_angle = np.array(joint_angle).reshape(4,3)
    # initial_q[:, 1:4] = joint_angle
    # initial_q = initial_q.flatten()

    filename = write_urdf(F1, F2, F3, F4)
    
    a = '0.000000000000000000e+00 -3.411167465433517232e-01 -1.445272339419535201e-01 3.792201907874313827e-02 0.000000000000000000e+00 0.000000000000000000e+00 -8.986635843187198436e-01 3.340959358925662537e-01 -3.143362829168838690e-01 0.000000000000000000e+00 0.000000000000000000e+00 -8.986635843187198436e-01 3.340959358925662537e-01 -3.143362829168838690e-01 0.000000000000000000e+00 0.000000000000000000e+00 -3.411167465433517232e-01 -1.445272339419535201e-01 3.792201907874313827e-02 0.000000000000000000e+00'
    initial_q = np.array(list(map(float, a.split())))

    # check_urdf_can_stand(filename, initial_q)
