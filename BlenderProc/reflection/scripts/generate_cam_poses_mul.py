"""
Script to adjust camera positions by moving them backwards along their direction vector.
It reads initial camera poses from a text file, applies position adjustments, and saves
the modified poses to a new text file.

NOTE: Currently uses Euler angles.
"""

import blenderproc as bproc
import numpy as np

#This cript assumes that training poses are present in a file:
#In future this logic needs to be written based on quaternions. 

camera_poses_txt_file = "reflection/resources/cam_poses.txt"
camera_poses_text_path = "reflection/resources/cam_poses_mul.txt"

with open(camera_poses_txt_file, "r") as f:
    poses_lines = f.readlines()

#Assuming that there are only 3 camera poses. 2nd pose is from front


test_poses = [] 

for index in range(3):
    line = [float(x) for x in poses_lines[index].split(',')]
    position_1, euler_rotation_1 = line[:3], line[3:6]

    def move_back(pos, magnitude):
        """
        Move a position vector backwards along its normalized direction by a specified magnitude.

        Args:
            pos (list or array): A 3D position [x, y, z].
            magnitude (float): Distance to move backwards.

        Returns:
            list: New 3D position after moving backwards.
        """
        input_position = np.array(pos, np.float32)
        input_position = np.reshape(input_position, (3,1))
        norm = np.linalg.norm(input_position)
        dir = input_position/norm
        new_pos = input_position + magnitude * dir
        new_pos = new_pos.flatten()
        return new_pos.tolist()
    
    if index < 2:
        new_pos = move_back(position_1, magnitude=2.25)
    else:
        new_pos = move_back(position_1, magnitude=2.4)
    new_pose = f"{new_pos[0]},{new_pos[1]},{new_pos[2]},{euler_rotation_1[0]},{euler_rotation_1[1]},{euler_rotation_1[2]}\n"

    test_poses.append(new_pose)

with open(camera_poses_text_path, "w") as f:
    for line in test_poses:
        f.write(f"{line}")