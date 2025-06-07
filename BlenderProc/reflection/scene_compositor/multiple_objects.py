import os
import sys
import glob
import json
import random
import datetime
import logging as log
import blenderproc as bproc


# Constants for random offsets
RAND_POS_Y_OFFSET = [-0.3, 0.3]

def sample_second_obj_position(obj):
    """
    Samples a random position for the second object, either to the left or right of the first object.
    """
    #Select left or right
    side = random.choice(["left", "right"])
    if side=="left":
        rand_x = random.uniform(-0.75, -0.35)
    else:
        rand_x = random.uniform(0.35, 0.75)
    
    rand_y = random.uniform(RAND_POS_Y_OFFSET[0], RAND_POS_Y_OFFSET[1])
    obj.set_location([rand_x,rand_y, obj.get_location()[2]])

def place_obj_without_collision(objs, collision_objs):
    """
    Places an object without collision with other objects.
    """
    status = bproc.object.sample_poses(
        objs,
        sample_pose_func=sample_second_obj_position,
        objects_to_check_collisions=collision_objs,
        max_tries = 500
    )
    key = list(status.keys())[0]
    return status[key][1]

def load_abo_paired_data(file_path="reflection/resources/abo_paired.json"):
    """
    Loads paired data for ABO objects from a JSON file. In future, this can be extended to load other datasets.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_second_object(first_obj_token, paired_data, data_dir):
    """
    Retrieves the second object paired with the first object based on the ABO dataset.
    """
    first_obj_token = str(first_obj_token)
    #Step 1 if it's abo dataset or not. TO DO : We can remove this later.
    if first_obj_token not in paired_data.keys():
        log.error(f"{str(first_obj_token)} This is not ABO data.")
        sys.exit(-1)

    second_obj_path = None
    #Step 2. Check if paired data is None or some valid value.
    second_obj_uid = paired_data[first_obj_token]
    if second_obj_uid is None:
        log.info(f"{str(first_obj_token)} : There is no paired data for this object")
        return second_obj_uid, second_obj_path

    if second_obj_uid not in paired_data.keys():
        log.error(f"{str(second_obj_uid)} This uid should not come as it is not ABO data.")
        sys.exit(-1) 

    second_obj_path = glob.glob( os.path.join(data_dir, f"**/{second_obj_uid}.glb"),  recursive=True )

    return second_obj_uid, second_obj_path[0]

def write_txt_file_collision_multiple_objs(args, collision_set):
    """
    Writes the UIDs of objects with incorrect placement to a text file.
    """
    if len(collision_set) == 0:
        log.info("No data in collision set.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"collision_{timestamp}.txt"
    with open( os.path.join(args.output_dir, filename), 'w') as f:
        for uid in collision_set:
            f.write(f"{uid}\n")

def create_dummy_renders(output_dir, num_renders):
    """
    Creates dummy render files for objects with placement issues.
    """
    os.makedirs(output_dir)
    for file_id in range(num_renders):
        with open(os.path.join(output_dir, f"{file_id}.hdf5"), 'w') as file:
            pass