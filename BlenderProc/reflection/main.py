"""
main.py

Main entry point for the MirrorVerse rendering pipeline.

Responsibilities:
- Parses command-line arguments.
- Loads input data (objects, HDRIs, textures, camera poses).
- Coordinates rendering for multiple scenes.
- Handles mirror placement and object validation.
- Calls functions from all other modules (bproc_io, render_main, geometry, errors).

Main functions:
- main: Entry function triggered by CLI.
- bulk_process: Iterates over object list and triggers rendering per object.
"""


import blenderproc as bproc
import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
import glob
import time
from typing import List, Dict
import random
from loguru import logger as log
import importlib.util

from blenderproc.python.types.MeshObjectUtility import MeshObject, Entity

"""
This code identifies the current script's directory, infers the project root 
directory (assumed to be one level up), and ensures the project root is added 
to sys.path for module import access.
"""
current_script_dir = os.path.dirname(os.path.abspath(__file__))
log.info(f"Current script directory: {current_script_dir}")
# Assuming 'reflection' is directly inside the root project directory (where rerun.py is)
project_root_dir = os.path.dirname(current_script_dir) # This should be 'Blenderproc' directory
log.info(f"Project root directory: {project_root_dir}")
if project_root_dir not in sys.path:
    sys.path.append(project_root_dir)

from reflection.scene_compositor.multiple_objects import (
                                            sample_second_obj_position,
                                            place_obj_without_collision,
                                            load_abo_paired_data,
                                            get_second_object,
                                            write_txt_file_collision_multiple_objs,
                                            create_dummy_renders )   

from reflection.bproc_io import (
                                load_3d_obj,
                                load_scene, 
                                sample_hdri, 
                                sample_floor, 
                                load_floor_textures, 
                                delete_objs, 
                                is_processed, 
                                write_json_file)
from reflection.render_main import (
                                    sample_cams, 
                                    random_pose_indices, 
                                    scene_lighting, 
                                    init, 
                                    single_process, 
                                    create_special_test_set, 
                                    render_views)
from reflection.errors import SpuriousObjException, CollisionException, ErrorRecord
from reflection.errors import (
                            count_spurious_files)
from reflection.geometry import (
                            normalise_objs) 

def bulk_process(args, mirror: List[MeshObject], cam_poses: List[np.ndarray] = [], diff_mirrors_data: Dict = {}):

    data_dir = Path(args.input_dir)
    uid_list = []
    if args.split_file != "":
        with open(args.split_file, "r") as f:
            for line in f.readlines():
                uid_list.append(line.strip())
    uid_set = set(uid_list)

    spurious_uids = set()
    if args.spurious_file != "":
        with open(args.spurious_file, "r") as f:
            log_data = json.load(f)
            spurious_uids = set(log_data.get("400", {}))
        log.info(f"Spurious UIDs in {args.spurious_file}: {len(spurious_uids)}")
    
    if args.fast_testing:
        args.num_render = 1
    total_processed = 0
    start_time = time.time()
    do_exit = False

    hdri_list = glob.glob( os.path.join(args.hdri, "**/*.exr") )
    floor_textures = load_floor_textures(args.textures)
    
    # # Placeholder to process spurious files
    spurious_files = {}

    generic_error_recoder = ErrorRecord()

    if args.fast_testing:
        bproc.camera.set_resolution(128, 128)

    if args.multiple_objects:
        mul_obj_paired_data = load_abo_paired_data()
        uids_with_incorrect_placement = set()

    for obj_dir in data_dir.iterdir():
        if not obj_dir.is_dir():
            continue
        
        if do_exit:
            log.warning('process will restart...')
            break

        # Hide all mirrors. Select base mirror
        diff_mirrors_data.hide_all_mirrors()
        if not args.small_mirrors:
            diff_mirrors_data.select_mirror(only_base_mirror=True)

        for file in obj_dir.glob(f"**/*.{args.model_3d_type}"):
            if (args.split_file != "" and file.stem not in uid_set) or file.stem in spurious_uids:
                continue

            #Pre-processing for multiple objects.
            if args.multiple_objects:
                second_obj_uid, second_obj_path = get_second_object(file.stem, mul_obj_paired_data, args.input_dir)
                if second_obj_uid is None:
                    #Don't process this. 
                    continue
            
            if args.multiple_objects:
                out_dir = Path(args.output_dir) / obj_dir.name / Path(f"{str(file.stem)}_{second_obj_uid}")
            else:
                out_dir = Path(args.output_dir) / obj_dir.name / file.stem
            if not args.reprocess and is_processed(out_dir, args.num_render):
                log.warning(f'{file.stem} is already processed with {args.num_render} renderings. Skipping...')
                continue
            if (
                total_processed >= args.max_objects
                or (time.time() - start_time) > args.max_time * 60
            ):
                do_exit = True
                break
            # Clear all key frames from the previous run
            bproc.utility.reset_keyframes()

            log.info(f"Processing {file.stem}")

            scene_type, hdri_info = sample_hdri(args, hdri_list)
            
            # Sample Floor texture
            floor_obj = bproc.filter.one_by_attr(mirror, "name", "floor")
            floor_texture_info = sample_floor(floor_obj, floor_textures[scene_type] ) 
            
            # Metadata
            metadata_ = {'hdri' : hdri_info, 'floor_texture' : floor_texture_info}

            objs = None
            second_objs = None
            try:
                cam_poses = sample_cams(cam_poses)
                if args.small_mirrors:
                    diff_mirrors_data.select_mirror(only_base_mirror=False)
                import_start = time.perf_counter()

                objs = load_3d_obj(str(file), use_legacy_obj_import=True, global_category_id=2, check_spurious=args.check_spurious)

                import_end = time.perf_counter()
                import_time = import_end - import_start

                # TODO fix this. log to spurious file with different error code and check.
                # if import_time > 60:
                #     log.warning(f"Importing {file.stem} took {import_time} seconds. Skipping large import files.")
                #     delete_objs(objs)
                #     continue
                if not args.multiple_objects:    
                    par_obj = normalise_objs(args, objs)
                else:
                    par_obj = normalise_objs(args, objs, rot_angle=0, offset_x=0, offset_y=0)
                    second_objs = load_3d_obj(second_obj_path, use_legacy_obj_import=True, global_category_id=3, check_spurious=args.check_spurious)
                    par_second_obj = normalise_objs(args, second_objs)
                    is_placed_correctly = place_obj_without_collision(second_objs, objs)
                    if not is_placed_correctly:
                        uids_with_incorrect_placement.add(file.stem)
                        raise CollisionException(f"{second_obj_uid} collided with {file.stem}",) 

                # if object is spurious, there will be an exception

                func_start = time.perf_counter()
                render_views(args, out_dir, metadata_)
                func_end = time.perf_counter()
                render_time = func_end - func_start
            
                if import_time < 10 and render_time > args.max_render_time:
                    do_exit = True
                    break
            except SpuriousObjException as e:
                log.error(f"Error processing {file.stem}: {e}. Error Code: {e.error_code}")
                spurious_uids.add(file.stem)
            except CollisionException as e:
                 log.error(f"Multiple Object Mode : {e}")
                 #Create dummy renders and proceed
                 create_dummy_renders(out_dir, args.num_render)
            except Exception as e:
                log.error(f"Error processing {file.stem}: {e}")
                # Write spurious file uid in a dictionary
                generic_error_recoder.write_generic_error(spurious_files, f"{e}", file.stem)

            delete_objs(objs)
            if args.multiple_objects:
                delete_objs(second_objs)
            total_processed += 1

            if args.small_mirrors:
                diff_mirrors_data.hide_all_mirrors()           

    # add the spurious uids to log file
    spurious_files["400"] = list(spurious_uids)
    log.info(f"Total Processed Files : {total_processed}\nSpurious Objects Found : {count_spurious_files(spurious_files)}")

    # Save json file of spurious objects
    spurious_files["GENERIC_ERROR_CODE"] = generic_error_recoder.get_error_codes()
    split_num = 0
    if args.split_file != "":
        split_num = args.split_file.split("/")[-1].split(".")[0].split("_")[-1]
    spurious_path = os.path.join(args.output_dir, f"spurious_{split_num}.json")
    log.info(f'writing spurious objects to: {spurious_path}')
    write_json_file(spurious_files, spurious_path)
    
    if args.multiple_objects:
        write_txt_file_collision_multiple_objs(args, uids_with_incorrect_placement)
    if not do_exit:
        log.info("Process completed successfully.")

def main(args):

    init(args)

    # load the mirror
    mirror, diff_mirrors_data = load_scene(args.mirror, mirror=True)

    scene_lighting()

    # sample cam poses
    all_cam_poses = []
    cam_poses = []
    if args.camera:
        # read the camera positions file and convert into homogeneous camera-world transformation
        with open(args.camera, "r") as f:
            for line in f.readlines():
                line = [float(x) for x in line.split(',')]
                position, euler_rotation = line[:3], line[3:6]
                matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
                all_cam_poses.append(matrix_world)

    # random sample args.cam poses out of the given cam_poses list
    # TODO: add this to bulk_process inner loop for each object instead of a group for more variation
    selected_indices = random_pose_indices(len(all_cam_poses), args.num_render)
    for index in selected_indices:
        cam_poses.append(all_cam_poses[index])    

    if args.create_rotate_trans_test_set:
        log.info("Special Mode to test rotation and depth capabitlites of the trained model.")
        create_special_test_set(args, mirror, cam_poses, diff_mirrors_data)
        return 1 #Return from here. No further processing.

    if args.fast_testing:
        cam_poses = [cam_poses[0]] # Only render 1st view for fast testing.

    if args.single_run:
        single_process(args, mirror, cam_poses) # for testing on a single object
    else:
        bulk_process(args, mirror, cam_poses, diff_mirrors_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        nargs="?",
        help="Path to the camera file, should be reflection/resources/cam_poses.txt",
        default="reflection/resources/cam_poses.txt",
    )
    parser.add_argument(
        "--mirror",
        help="Path to the scene.blend mirror file, should be reflection/resources/base_mirror.glb",
        nargs="?",
        default="reflection/resources/all_mirrors.glb",
    )
    parser.add_argument(
        "--hdri",
        help="Path to the hdri folder",
        nargs="?",
        default="/data/manan/data/objaverse/blenderproc/resources/HDRI",
    )
    parser.add_argument(
        "--textures",
        help="Path to the textures folder",
        nargs="?",
        default="blenderproc/resources/textures",
    )   
    parser.add_argument(
        "--object",
        help="Path to the objaverse.glb file, could be reflection/resources/objaverse_examples/6f99fb8c2f1a4252b986ed5a765e1db9/6f99fb8c2f1a4252b986ed5a765e1db9.glb",
        nargs="?",
        default="reflection/resources/objaverse_examples/063b1b7d877a402ead76cedb06341681/063b1b7d877a402ead76cedb06341681.glb",
    )
    parser.add_argument(
        "--input_dir",
        help="Path to the objaverse dataset, could be reflection/resources/objaverse_examples/",
        nargs="?",
        default="reflection/resources/objaverse_examples/",
    )
    parser.add_argument(
        "--split_file",
        help="Path to the split file, should be in reflection/resources/splits/",
        nargs="?",
        default="",
    )
    parser.add_argument("--num_render", type=int, help="Number of renderings per object", nargs="?", default=3)
    parser.add_argument(
        "--spurious_file",
        help="Path to the spurious objects file, could be reflection/resources/spurious_0.json",
        nargs="?",
        default="/data/manan/data/objaverse/blenderproc/hf-objaverse-v1/spurious_0.json",
    )
    parser.add_argument(
        "--output_dir",
        nargs="?",
        help="Path to where the final files, will be saved. could be reflection/output",
        default="reflection/output/blenderproc",
    )
    parser.add_argument(
        "--max_objects",
        nargs="?",
        type=int,
        help="Max objects to process in this run. (default: 180, considering 30min and 10s per object)",
        default=75, # found to be good
    )
    parser.add_argument(
        "--max_time",
        nargs="?",
        type=int,
        help="Max time for this run (in mins). (default: 30min)",
        default=30,
    )
    parser.add_argument(
        "--max_render_time",
        nargs="?",
        type=int,
        help="Max render time before the process stops. (default: 30s)",
        default=30,
    )
    parser.add_argument(
        "--model_3d_type",
        nargs="?",
        type=str,
        help="file-type of 3D model glb, obj, fbx . (default: glb)",
        default="glb",
    )
    parser.add_argument(
        "--small_mirrors",
        action='store_true', default=False,
        help="Enable it choose randomly from small mirrors.",
    )
    parser.add_argument(
        "--disable_rotate",
        action="store_true",
        default=False,
        help="disable rotating objects. Use this for certain uids from objaverse.",
    )
    parser.add_argument(
        "--fast_testing",
        action='store_true', default=False,
        help="Enable it so that changes can be tested quickly. Use degraded options",
    )
    parser.add_argument(
        "--single_run", action="store_true", help="Whether to use single run for testing"
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Whether to reprocess same object. This does not check `is_processed` and appends to same output folder. NOTE: Do not use this with `rerun.py`.",
    )
    parser.add_argument(
        "--check_spurious",
        action="store_true",
        help="Whether to check for spurious object when importing. Add this option if all spurious objects are not known. Else pass the complete generated spurious file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed for reproducible rendering",
        default=None,
    )
    parser.add_argument(
        "--multiple_objects",
        action="store_true",
        help="Renders multiple objects",
        default=False,
    )
    parser.add_argument(
        "--create_rotate_trans_test_set",
        action="store_true",
        help="This is a special flag. If this flag is set, only test-set for testing rotation and depth capabilities of the model will be tested."
    )

    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)
        os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.seed)
    main(args)
