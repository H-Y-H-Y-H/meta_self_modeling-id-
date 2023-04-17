import os
import xml.etree.ElementTree as ET

for robot_name in os.listdir():
    if '.' in robot_name:
        continue
    tree = ET.parse(os.path.join(robot_name, f"{robot_name}.urdf"))
    for link in tree.findall('link'):
        link.find('inertial').find('mass').set('value', '0')
        for col in link.findall('collision'):
            link.remove(col)

    for elem in tree.iter():
        if elem.tag == 'color':
            color = elem.attrib['rgba'].split()
            color[3] = '0.5'
            color = ' '.join(color)
            elem.set('rgba', color)

    tree.write(os.path.join(robot_name, f"{robot_name}_nocol.urdf"))