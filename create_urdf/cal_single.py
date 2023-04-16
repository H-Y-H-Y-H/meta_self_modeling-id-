from math import sqrt, cos, sin, pi
import pybullet as p
import numpy as np
import urdfpy as upy
import os

# Some global parameters
# The icosahedron is split into four layers.
# face_interval: In each layer, there are five faces. Their internal angle is 72°.
# face_z: The first rotation between the base and the link along the z axis
face_interval = (72 / 180) * np.pi
alfas = [(90 - 52.62) / 180 * np.pi, (90 - 10.81) / 180 * np.pi,
         (90 + 10.81) / 180 * np.pi, (90 + 52.62) / 180 * np.pi]
betas = [(0 / 180) * np.pi, (0 / 180) * np.pi,
         (0 / 180) * np.pi, (0 / 180) * np.pi]
gammas = [(0 / 180) * np.pi, (0 / 180) * np.pi,
          (36 / 180) * np.pi, (36 / 180) * np.pi]
face_z = (0 / 180) * np.pi

save_pth = '../data/robot_urdf_search/'
# save_data_pth = '../data/robot_warehouse/'

'''
Description: Get the xyz, rpy of URDF from face id.
Creater: Yuhang Hu
Input:
    leg_id: The id of the joint between face and leg
    seed: Random seed
Ouput:
    xyz: The xyz of joint in URDF
    rpy: The rpy of joint in URDF
'''




def get_xyz_rpy(leg_id, firstjoint):
    # The layer of the base_leg_joint, top = 0, bottom = 3
    leg_layer = leg_id // 5

    alfa = alfas[leg_layer]
    beta = betas[leg_layer]
    gamma = gammas[leg_layer] + face_interval * (leg_id % 5)
    face_z = firstjoint * np.pi * (30 / 180)
    # print(f"The angles are {alfa, beta, gamma,face_z}")

    rot_x = [[1, 0, 0, 0],
             [0, np.cos(alfa), -np.sin(alfa), 0],
             [0, np.sin(alfa), np.cos(alfa), 0],
             [0, 0, 0, 1]]

    rot_y = [[np.cos(beta), 0, np.sin(beta), 0],
             [0, 1, 0, 0],
             [-np.sin(beta), 0, np.cos(beta), 0],
             [0, 0, 0, 1]]

    rot_z = [[np.cos(gamma), -np.sin(gamma), 0, 0],
             [np.sin(gamma), np.cos(gamma), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]

    rot_face_z = [[np.cos(face_z), -np.sin(face_z), 0, 0],
                  [np.sin(face_z), np.cos(face_z), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]

    # after rotation at the body origin,
    # translating the joint from body origin to 20 faces along z axis.
    # This distance should be all the same becasue the body origin is at the
    # center of geometry.
    Trans = [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0.08940753778625156],
             [0, 0, 0, 1]]

    rot_z = np.asarray(rot_z)
    rot_y = np.asarray(rot_y)
    rot_x = np.asarray(rot_x)
    rot_face_z = np.asarray(rot_face_z)
    Trans = np.asarray(Trans)

    R1 = np.dot(rot_z, rot_y)
    R2 = np.dot(R1, rot_x)
    R3 = np.dot(R2, rot_face_z)
    T = np.dot(R3, Trans)
    xyz_rpy = upy.matrix_to_xyz_rpy(T)

    # convert array to list
    xyz = [xyz_rpy[i].tolist() for i in range(3)]
    # xyz = [subitem for item in xyz for subitem in item]
    rpy = [xyz_rpy[i] for i in range(3, 6)]

    return [xyz, rpy]


# [xyz, rpy] = get_xyz_rpy(F1[0], F1[1])


sim_ixy = [-2.8331E-19, 5.0822E-21, 4.6587E-21,
           5.034E-21, 1.6941E-21, 4.6587E-21, -2.1949E-12, -4.6587E-21, -1.2705E-21, 4.7646E-21, -3.3881E-21, 3.8116E-21]
sim_ixz = [-1.865E-12, -1.2705E-21, -8.4703E-21,
           8.4703E-22, 2.5411E-21, -5.5057E-21, -1.6612E-12, -1.6941E-21, -8.4703E-22, -4.2352E-21, 4.2352E-21, -1.2705E-21]
sim_iyz = [-2.234E-16, 2.1165E-05, -1.2705E-21,
           4.0028E-15, 2.1165E-05, -3.3881E-21, 3.854E-19, -1.6941E-21, -3.8116E-21, -3.8065E-14, -8.4703E-22, -3.3881E-21]


ixx = [3.2684E-05, 3.2684E-05, 3.2684E-05, 3.2684E-05,
       3.2241E-05, 3.2684E-05, 3.2684E-05, 3.2684E-05, 3.2684E-05, 3.2684E-05, 3.2684E-05, 3.2684E-05]
ixy = [-2.5662E-09, -2.5662E-09, -2.5662E-09, -2.5662E-09,
       3.5298E-07, -2.5662E-09, -2.5662E-09, -2.5662E-09, -2.5662E-09, -2.5662E-09, -2.5662E-09, -2.5662E-09]
ixz = [2.5662E-09, 2.5662E-09, 2.5662E-09, 2.5662E-09,
       2.0781E-07, 2.5662E-09, 2.5662E-09, 2.5662E-09, 2.5662E-09, 2.5662E-09, 2.5662E-09, 2.5662E-09]
iyy = [9.3595E-05, 9.1229E-05, 8.6498E-05, 8.4132E-05,
       8.5969E-05, 9.1229E-05, 9.3595E-05, 9.1229E-05, 8.6498E-05, 8.4132E-05, 8.6498E-05, 9.1229E-05]
iyz = [-2.1275E-13, 4.0975E-06, 4.0975E-06, -2.1275E-13,
       -3.9192E-06, -4.0975E-06, -2.1275E-13, 4.0975E-06, 4.0975E-06, -2.1275E-13, -4.0975E-06, -4.0975E-06]
izz = [7.512E-05, 7.7485E-05, 8.2217E-05, 8.4583E-05,
       8.1482E-05, 7.7485E-05, 7.512E-05, 7.7485E-05, 8.2217E-05, 8.4583E-05, 8.2217E-05, 7.7485E-05]


def leg_urdf(a, b, c, d, f):
    change = [3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2]
    b = change[b]
    F1 = [a, b, c, d]
    [xyz, rpy] = get_xyz_rpy(F1[0], F1[1])
    rad1 = F1[2] * 30 * np.pi / 180
    rad2 = F1[3] * 30 * np.pi / 180
    # print(rad1, rad2)
    line2 = [
        '  <link',
        '   name="%s0">' % (F1[0]),
        '   <inertial>',
        '      <origin',
        '        xyz="0 0 0.028508"',
        '        rpy="0 0 0" />',
        '      <mass',
        '        value="0.076173" />',
        '      <inertia',
        '        ixx="3.054E-05"',
        '        ixy="%f"' % (sim_ixy[F1[1]]),
        '        ixz="%f"' % (sim_ixz[F1[1]]),
        '        iyy="2.1077E-05"',
        '        iyz="%f"' % (sim_iyz[F1[1]]),
        '        izz="2.1165E-05" />',
        '    </inertial>',
        '    <visual>',
        '      <origin',
        '        xyz="0 0 0"',
        '        rpy="0 0 0" />',
        '      <geometry>',
        '        <mesh',
        '          filename="package://meshes/1.STL" />',
        '      </geometry>',
        '      <material',
        '        name="">',
        '        <color',
        '          rgba="1 1 1 1" />',
        '      </material>',
        '    </visual>',
        '    <collision>',
        '      <origin',
        '        xyz="0 0 0"',
        '        rpy="0 0 0" />',
        '      <geometry>',
        '        <mesh',
        '          filename="package://meshes/1.STL" />',
        '      </geometry>',
        '    </collision>',
        '  </link>',
        '  <joint',
        '    name="body_%s0"' % F1[0],
        '    type="fixed">',
        '    <origin',
        '        xyz="%f %f %f"' % (
            xyz[0], xyz[1], xyz[2]),
        '        rpy="%f %f %f" />' % (rpy[0], rpy[1], rpy[2]),
        '    <parent',
        '      link="base_link" />',
        '    <child',
        '      link="%s0" />' % F1[0],
        '    <axis',
        '      xyz="0 0 0" />',
        '  </joint>',
        '  <link',
        '    name="%s1">' % (F1[0]),
        '    <inertial>',
        '      <origin',
        '        xyz="0.060018 -1.4108E-06 1.4108E-06"',
        '        rpy="0 0 0" />',
        '    <mass',
        '        value="0.10689" />',
        '      <inertia',
        '        ixx="%s"' % (ixx[F1[2]]),
        '        ixy="%s"' % (ixy[F1[2]]),
        '        ixz="%s"' % (ixz[F1[2]]),
        '        iyy="%s"' % (iyy[F1[2]]),
        '        iyz="%s"' % (iyz[F1[2]]),
        '        izz="%s" />' % (izz[F1[2]]),
        '    </inertial>',
        '    <visual>',
        '      <origin',
        '        xyz="0 0 0"',
        '        rpy="0 0 0" />',
        '      <geometry>',
        '        <mesh',
        # edit
        '          filename="package://meshes/module%s.STL" />' % (
            F1[2]),
        '      </geometry>',
        '      <material',
        '        name="">',
        '        <color',
        '          rgba="0.79216 0.81961 0.93333 1" />',
        '      </material>',
        '    </visual>',
        '    <collision>',
        '      <origin',
        '        xyz="0 0 0"',
        '        rpy="0 0 0" />',
        '      <geometry>',
        '        <mesh',
        # edit
        '          filename="package://meshes/module%s.STL" />' % (
            F1[2]),
        '      </geometry>',
        '    </collision>',
        '  </link>',
        '  <joint',
        '    name="%s0_%s1"' % (F1[0], F1[0]),
        '    type="revolute">',
        '    <origin',
        '      xyz="0 0 0.04"',
        '      rpy="-1.558 -1.5708 -0.012835" />',
        '    <parent',
        '      link="%s0" />' % (F1[0]),
        '    <child',
        '      link="%s1" />' % (F1[0]),
        '    <axis',
        '      xyz="0 0 1" />',
        '    <limit',
        '      lower="-6.28"',
        '      upper="6.28"',
        '      effort="10"',
        '      velocity="100" />',
        '  </joint>',
        '  <link',
        '    name="%s2">' % (F1[0]),
        '    <inertial>',
        '      <origin',
        '        xyz="0.060018 -1.4108E-06 1.4108E-06"',
        '        rpy="0 0 0" />',
        '    <mass',
        '        value="0.10689" />',
        '      <inertia',
        '        ixx="%s"' % (ixx[F1[3]]),
        '        ixy="%s"' % (ixy[F1[3]]),
        '        ixz="%s"' % (ixz[F1[3]]),
        '        iyy="%s"' % (iyy[F1[3]]),
        '        iyz="%s"' % (iyz[F1[3]]),
        '        izz="%s" />' % (izz[F1[3]]),
        '    </inertial>',
        '    <visual>',
        '      <origin',
        '        xyz="0 0 0"',
        '        rpy="0 0 0" />',
        '      <geometry>',
        '        <mesh',
        # edit
        '          filename="package://meshes/module%s.STL" />' % F1[3],
        '      </geometry>',
        '      <material',
        '        name="">',
        '        <color',
        '          rgba="0.79216 0.81961 0.93333 1" />',
        '      </material>',
        '    </visual>',
        '    <collision>',
        '      <origin',
        '        xyz="0 0 0"',
        '        rpy="0 0 0" />',
        '      <geometry>',
        '        <mesh',
        # edit
        '          filename="package://meshes/module%s.STL" />' % F1[3],
        '      </geometry>',
        '    </collision>',
        '  </link>',
        '  <joint',
        '    name="%s1_%s2"' % (F1[0], F1[0]),
        '    type="revolute">',
        '    <origin',
        '      xyz="0.0816 0 0"',
        '      rpy="%f 0 0" />' % (rad1),
        '    <parent',
        '      link="%s1" />' % (F1[0]),
        '    <child',
        '      link="%s2" />' % (F1[0]),
        '    <axis',
        '      xyz="0 0 1" />',
        '    <limit',
        '      lower="-6.28"',
        '      upper="6.28"',
        '      effort="10"',
        '      velocity="100" />',
        '  </joint>',
        '  <link',
        '    name="%s3">' % (F1[0]),
        '    <inertial>',
        '      <origin',
        '        xyz="0.051733 -0.042281 -1.3878E-17"',
        '        rpy="0 0 0" />',
        '    <mass',
        '        value="0.12059" />',
        '      <inertia',
        '        ixx="0.00025425"',
        '        ixy="4.8319E-05"',
        '        ixz="0"',
        '        iyy="5.0194E-05"',
        '        iyz="-3.3141E-21"',
        '        izz="0.00027482" />',
        '    </inertial>',
        '    <visual>',
        '      <origin',
        '        xyz="0 0 0"',
        '        rpy="0 0 0" />',
        '      <geometry>',
        '        <mesh',
        '          filename="package://meshes/leg.STL" />',  # edit
        '      </geometry>',
        '      <material',
        '        name="">',
        '        <color',
        '          rgba="0.79216 0.81961 0.93333 1" />',
        '      </material>',
        '    </visual>',
        '    <collision>',
        '      <origin',
        '        xyz="0 0 0"',
        '        rpy="0 0 0" />',
        '      <geometry>',
        '        <mesh',
        '          filename="package://meshes/leg.STL" />',  # edit
        '      </geometry>',
        '    </collision>',
        '  </link>',
        '  <joint',
        '    name="%s2_%s3"' % (F1[0], F1[0]),
        '    type="revolute">',
        '    <origin',
        '      xyz="0.0816 0 0"',
        '      rpy="%f 0 0" />' % (rad2),
        '    <parent',
        '      link="%s2" />' % (F1[0]),
        '    <child',
        '      link="%s3" />' % (F1[0]),
        '    <axis',
        '      xyz="0 0 1" />',
        '    <limit',
        '      lower="-6.28"',
        '      upper="6.28"',
        '      effort="10"',
        '      velocity="100" />',
        '  </joint>']
    for line in line2:
        f.write(line)
        f.write('\n')


def write_urdf(a, b, c, d, save_data_pth):
    str = "%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
        a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], d[0], d[1], d[2], d[3])
    filename = str + "/" + str + ".urdf"
    os.makedirs(os.path.dirname(save_data_pth + filename), exist_ok=True)
    with open(save_data_pth + filename, "w") as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        f.write('<robot\n')
        f.write('  name="mod_test">\n')
        f.write('  <link\n')
        f.write('    name="base_link">\n')
        f.write('    <inertial>\n')
        f.write('      <origin\n')
        f.write(
            '        xyz="-1.5226495220944E-06 1.5812928362389E-07 -5.05247252856584E-07"\n')
        f.write('        rpy="0 0 0" />\n')
        f.write('      <mass\n')
        f.write('        value="0.422806992949252" />\n')
        f.write('      <inertia\n')
        lines = ['        ixx="0.00237499139253511"',
                 '        ixy="1.5717831530364E-08"',
                 '        ixz="1.49286265366169E-08"',
                 '        iyy="0.00237502154661776"',
                 '        iyz="-2.02154172619351E-08"',
                 '        izz="0.00237502595466009" />',
                 '    </inertial>',
                 '    <visual>',
                 '      <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '      <geometry>',
                 '        <mesh',
                 '          filename="package://meshes/base_link.STL" />',
                 '      </geometry>',
                 '      <material',
                 '        name="">',
                 '        <color',
                 '          rgba="0.498039215686275 0.498039215686275 0.498039215686275 1" />',
                 '      </material>',
                 '    </visual>',
                 '    <collision>',
                 '      <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '      <geometry>',
                 '        <mesh',
                 '          filename="package://meshes/base_link.STL" />',
                 '      </geometry>',
                 '    </collision>',
                 '  </link>']
        for line in lines:
            f.write(line)
            f.write('\n')
        leg_urdf(a[0], a[1], a[2], a[3], f)
        leg_urdf(b[0], b[1], b[2], b[3], f)
        leg_urdf(c[0], c[1], c[2], c[3], f)
        leg_urdf(d[0], d[1], d[2], d[3], f)

        line3 = ['  <link',
                 '    name = "ball_1">',
                 '    <inertial>',
                 '    <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '    <mass',
                 '        value="0" />',
                 '      <inertia',
                 '        ixx="0"',
                 '        ixy="0"',
                 '        ixz="0"',
                 '        iyy="0"',
                 '        iyz="0"',
                 '        izz="0" />',
                 '    </inertial>'
                 '    <visual>',
                 '      <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '     <geometry>',
                 '        <sphere radius = "0.01"/>',
                 '      </geometry>',
                 '      <material',
                 '        name="green">',
                 '        <color',
                 '          rgba="0.22 1 0.4 1" />',
                 '      </material>',
                 '    </visual>',
                 '  </link>',
                 '  <joint',
                 '  name="%s3_ball1"' % a[0],
                 '  type="fixed">',
                 '    <origin',
                 '      xyz="0.0612 -0.13431 0"',
                 '      rpy="0 0 0" />',
                 '    <parent',
                 '      link="%s3" />' % a[0],
                 '    <child',
                 '      link="ball_1" />',
                 '    <axis',
                 '      xyz="0 0 0" />',
                 '  </joint>',
                 '  <link',
                 '    name = "ball_2">',
                 '    <inertial>',
                 '    <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '    <mass',
                 '        value="0" />',
                 '      <inertia',
                 '        ixx="0"',
                 '        ixy="0"',
                 '        ixz="0"',
                 '        iyy="0"',
                 '        iyz="0"',
                 '        izz="0" />',
                 '    </inertial>'
                 '    <visual>',
                 '      <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '      <geometry>',
                 '        <sphere radius = "0.01"/>',
                 '      </geometry>',
                 '      <material',
                 '        name="blue">',
                 '        <color',
                 '          rgba="0.16 0.52 0.8 1" />',
                 '      </material>',
                 '    </visual>',
                 '  </link>',
                 '  <joint',
                 '  name="%s3_ball2"' % b[0],
                 '  type="fixed">',
                 '    <origin',
                 '      xyz="0.0612 -0.13431 0"',
                 '      rpy="0 0 0" />',
                 '    <parent',
                 '      link="%s3" />' % b[0],
                 '    <child',
                 '      link="ball_2" />',
                 '    <axis',
                 '      xyz="0 0 0" />',
                 '  </joint>']

        for line in line3:
            f.write(line)
            f.write('\n')

        line4 = ['  <link',
                 '    name = "ball_3">',
                 '    <inertial>',
                 '    <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '    <mass',
                 '        value="0" />',
                 '      <inertia',
                 '        ixx="0"',
                 '        ixy="0"',
                 '        ixz="0"',
                 '        iyy="0"',
                 '        iyz="0"',
                 '        izz="0" />',
                 '    </inertial>'
                 '    <visual>',
                 '      <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '     <geometry>',
                 '        <sphere radius = "0.01"/>',
                 '      </geometry>',
                 '      <material',
                 '        name="blue">',
                 '        <color',
                 '          rgba="0.16 0.52 0.8 1" />',
                 '      </material>',
                 '    </visual>',
                 '  </link>',
                 '  <joint',
                 '  name="%s3_ball3"' % c[0],
                 '  type="fixed">',
                 '    <origin',
                 '      xyz="0.0612 -0.13431 0"',
                 '      rpy="0 0 0" />',
                 '    <parent',
                 '      link="%s3" />' % c[0],
                 '    <child',
                 '      link="ball_3" />',
                 '    <axis',
                 '      xyz="0 0 0" />',
                 '  </joint>',
                 '  <link',
                 '    name = "ball_4">',
                 '    <inertial>',
                 '    <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '    <mass',
                 '        value="0" />',
                 '      <inertia',
                 '        ixx="0"',
                 '        ixy="0"',
                 '        ixz="0"',
                 '        iyy="0"',
                 '        iyz="0"',
                 '        izz="0" />',
                 '    </inertial>'
                 '    <visual>',
                 '      <origin',
                 '        xyz="0 0 0"',
                 '        rpy="0 0 0" />',
                 '      <geometry>',
                 '        <sphere radius = "0.01"/>',
                 '      </geometry>',
                 '      <material',
                 '        name="yellow">',
                 '        <color',
                 '          rgba="1 1 0 1" />',
                 '      </material>',
                 '    </visual>',
                 '  </link>',
                 '  <joint',
                 '  name="%s3_ball4"' % d[0],
                 '  type="fixed">',
                 '    <origin',
                 '      xyz="0.0612 -0.13431 0"',
                 '      rpy="0 0 0" />',
                 '    <parent',
                 '      link="%s3" />' % d[0],
                 '    <child',
                 '      link="ball_4" />',
                 '    <axis',
                 '      xyz="0 0 0" />',
                 '  </joint>']

        for line in line4:
            f.write(line)
            f.write('\n')

        f.write('</robot>')
        f.close()
    return filename.split('/')[0]

if __name__ == "__main__":
    F1 = [10, 1, 0, 1]
    F2 = [11, 0, 0, 0]
    F3 = [13, 1, 0, 1]
    F4 = [14, 0, 0, 0]
    #  up down change, 3 and 9 no change

    filename = write_urdf(F1, F2, F3, F4, save_pth)

