import blenderproc as bproc

#This cript assumes that training poses are present in a file:
#In future this logic needs to be written based on quaternions. 

camera_poses_txt_file = "reflection/resources/cam_poses.txt"
camera_poses_text_path = "reflection/resources/cam_poses_test.txt"

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

    def avg_pos(id_elem): 
        return (position_1[id_elem]+position_2[id_elem])/2
    
    def avg_euler(id_elem): 
        if euler_rotation_1[id_elem] >= 0:
            return (euler_rotation_1[id_elem]+euler_rotation_2[id_elem])/2
        else:
            return -(abs(euler_rotation_1[id_elem])+euler_rotation_2[id_elem])/2

    new_pose = f"{avg_pos(0)},{avg_pos(1)},{avg_pos(2)},{avg_euler(0)},{euler_rotation_1[1]},{avg_euler(2)}\n"

    test_poses.append(new_pose)

with open(camera_poses_text_path, "w") as f:
    for line in test_poses:
        f.write(f"{line}")