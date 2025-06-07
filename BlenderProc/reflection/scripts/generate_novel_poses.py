import blenderproc as bproc
import numpy as np
#This cript assumes that training poses are present in a file:
#In future this logic needs to be written based on quaternions. 

camera_poses_txt_file = "reflection/resources/cam_poses.txt"
camera_poses_text_path = "reflection/resources/cam_novel_poses.txt"

with open(camera_poses_txt_file, "r") as f:
    poses_lines = f.readlines()

#Assuming that there are only 3 camera poses. 2nd pose is from front


test_poses = [] 

for line in poses_lines:
    test_poses.append(line)

for index in range(2):
    line = [float(x) for x in poses_lines[index].split(',')]
    position_1, euler_rotation_1 = line[:3], line[3:6]

    line = [float(x) for x in poses_lines[2].split(',')]
    position_2, euler_rotation_2 = line[:3], line[3:6]

    def avg_pos(id_elem, t): 
        return position_1[id_elem] + t * (position_2[id_elem] - position_1[id_elem]) 
    
    def avg_euler(id_elem, t): 
        if euler_rotation_1[id_elem] >= 0:
            return euler_rotation_1[id_elem] + t * (euler_rotation_2[id_elem] - euler_rotation_1[id_elem]) 
        else:
            interp = abs(euler_rotation_1[id_elem]) + t * (euler_rotation_2[id_elem] - abs(euler_rotation_1[id_elem])) 
            return -interp

    step = 0.1
    for alpha in np.arange(step, 1.0-step, step):
        new_pose = f"{avg_pos(0,alpha)},{avg_pos(1,alpha)},{avg_pos(2,alpha)},{avg_euler(0,alpha)},{euler_rotation_1[1]},{avg_euler(2,alpha)}\n"
        test_poses.append(new_pose)

#Create the order  (hard-coded for 0.05 step)
# ordered_poses = [ test_poses[0] ]
# ordered_poses += test_poses[3:21]
# ordered_poses.append( test_poses[2]  )
# ordered_poses += test_poses[38:20:-1]
# ordered_poses.append(  test_poses[1]  )
# with open(camera_poses_text_path, "w") as f:
#     for line in ordered_poses:
#         f.write(f"{line}")

with open(camera_poses_text_path, "w") as f:
    for line in test_poses:
        f.write(f"{line}")