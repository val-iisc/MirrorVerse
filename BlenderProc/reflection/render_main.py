
import blenderproc as bproc
import math
import numpy as np
from pathlib import Path
import os
from loguru import logger as log
from typing import List, Dict
import random
import time
import sys
import glob

from blenderproc.python.types.MeshObjectUtility import MeshObject
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.camera.CameraUtility import rotation_from_forward_vec

from reflection.bproc_io import (
                            load_3d_obj, 
                            sample_hdri,
                            save_cam_states, 
                            sample_floor,
                            load_floor_textures, 
                            delete_objs, 
                            is_processed)
from reflection.geometry import (
                            normalise_objs)
'''
render_main.py

Responsible for rendering pipeline tasks: camera sampling, lighting, rendering, and test set generation.

Functions:
- init: Initializes BlenderProc scene and renderer.
- scene_lighting, create_light: Sets up lighting for the scene.
- sample_cams: Randomly samples valid camera poses.
- render_views: Renders images, depth maps, normal maps, and segmentation masks.
- create_special_test_set: Generates special evaluation renderings (e.g., for pose changes).
- random_pose_indices: Selects camera pose indices for test set.
- single_process: Combines object loading, rendering, and saving for one sample.
"""
'''
def create_light(name, light_type, energy, location, rotation, radius=0.25):
    light = bproc.types.Light(light_type=light_type, name=name)
    light.set_energy(energy)
    light.set_location(location)
    light.set_scale((1, 1, 1))
    light.set_rotation_euler(rotation)
    light.set_radius(radius)
    return light


def scene_lighting():
    # Rim/Back light
    rim_light = create_light(
        name="RimLight",
        light_type="AREA",
        energy=600,
        location=(0, 4, 5),
        rotation=(math.radians(-45), 0, 0),
    )
    rim_light.blender_obj.data.size = 2


def sample_cams(cam_poses: List[np.ndarray] = []):

    if len(cam_poses):
        for matrix_world in cam_poses:
            bproc.camera.add_camera_pose(matrix_world)
    else:
        # sample cameras
        look_at = np.array([0, 4, 0])
        cam_locations = [np.array([2, -5, 2]), np.array([-2, -5, 2])]
        for cam_location in cam_locations:
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                look_at - cam_location, up_axis="Z"
            )
            cam2world_matrix = bproc.math.build_transformation_mat(
                cam_location, rotation_matrix
            )
            cam_poses.append(cam2world_matrix)
            bproc.camera.add_camera_pose(cam2world_matrix)
    return cam_poses


def init(args):
    """Initialise the blenderproc settings and optionally override 
    from the ones in DefaultConfig.py.
    """
    bproc.init()
    bproc.renderer.set_light_bounces(
        diffuse_bounces=5, 
        glossy_bounces=5, 
        max_bounces=5
    )

    #enable_depth_output function can only be called once. Please be careful!!
    bproc.renderer.enable_depth_output(activate_antialiasing=False)




def render_views(args, out_dir: str, metadata : Dict):
    # activate normal map rendering. This needs to be called after add_camera_pose !!
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(default_values={"category_id": 0})
    data = bproc.renderer.render()
    data = save_cam_states(data, metadata)
    append_to_existing_output = True if args.reprocess else False
    if args.create_rotate_trans_test_set:
        append_to_existing_output = True
    bproc.writer.write_hdf5(out_dir, data, append_to_existing_output=append_to_existing_output)

def single_process(args, mirror: List[MeshObject], cam_poses: List[np.ndarray] = []):
    """LEGACY CODE. Modify and use if required"""
    cam_poses = sample_cams(cam_poses)
    hdri_list = os.listdir(args.hdri)
    sample_hdri(args, hdri_list)
    objs = load_3d_obj(args.object, global_category_id=2)
    par_obj = normalise_objs(args, objs)

    bproc.renderer.enable_segmentation_output(default_values={"category_id": 0})

    # render data
    data = bproc.renderer.render()
    data = save_cam_states(data, metadata={}) #We can fix this later.
    bproc.writer.write_hdf5(args.output_dir, data)

def random_pose_indices(num_cam_poses, num_renders):
    all_pose_indices = list(range(num_cam_poses))
    
    #Case : When number of cam poses are less than number of renders
    if num_cam_poses <= num_renders:
        return all_pose_indices
    
    selected_indices = []
    bucket_size = num_cam_poses // num_renders
    
    for r_id in range(num_renders):
        start = r_id * bucket_size
        end = min(num_cam_poses, start + bucket_size)
        if r_id == num_renders-1:
            end = num_cam_poses if end < num_cam_poses else end
        selected_indices.append(random.choice(all_pose_indices[start:end]))
    
    return selected_indices

def create_special_test_set(args, mirror, cam_poses, diff_mirrors_data):
    #Step 1: Select objects from objaverse and ABO.
    selected_uids = ["1a83213a144a4c19bb834cc348e45bfa", "3e0e44ca3833416096b819722641f721",
                      "4a4aee2d02e348c4a9a3b9df87441647", "4fc711dc5c8c4d2e8d56f2047e918973",
                      "5a1fb7f898324a6a97483e7a64684639", 
                      "B075NRDS91", "B075X1T4ZS", "B075X4J118", "B075X4PTS8", "B0828F62FS"]

    #Step 2: Check if these objects can be found in the input directory and collect the path
    full_paths = []
    for uid in selected_uids:
        f_paths = glob.glob( os.path.join(args.input_dir, f"**/{uid}.glb"), recursive=True)
        if len(f_paths) == 0:
            log.error(f"{uid} not found in {args.input_dir}")
            sys.exit(-1)
        full_paths.append(f_paths[0])

    start_time = time.time()
    do_exit = False

    hdri_list = glob.glob( os.path.join(args.hdri, "**/*.exr") )
    floor_textures = load_floor_textures(args.textures)
    
    # Hide all mirrors. Select base mirror
    diff_mirrors_data.hide_all_mirrors()
    diff_mirrors_data.select_mirror(only_base_mirror=True)

    #Settings for sampling ration for rotation and translation 
    NUM_PER_OBJECT_ROTATION_RENDERS = 5
    NUM_PER_OBJECT_TRANSLATION_RENDERS = 5
    

    def render_an_object(hdri_index, hdri_angle, floor_index, out_dir, model_path, uid, rotation_angle, offset_x, offset_y):
        # Clear all key frames from the previous run
        bproc.utility.reset_keyframes()

        scene_type, hdri_info = sample_hdri(args, hdri_list, index=hdri_index, angle=hdri_angle)
        
        # Sample Floor texture
        floor_obj = bproc.filter.one_by_attr(mirror, "name", "floor")
        floor_texture_info = sample_floor(floor_obj, floor_textures[scene_type], index=floor_index) 
        
        # Metadata
        metadata_ = {'hdri' : hdri_info, 'floor_texture' : floor_texture_info, 
                    'rot_angle':rotation_angle, 'offset_x': offset_x, 'offset_y': offset_y}

        
        objs = None
        try:
            cam_poses_local = sample_cams(cam_poses)
            import_start = time.perf_counter()

            objs = load_3d_obj(model_path, use_legacy_obj_import=True, global_category_id=2, check_spurious=args.check_spurious)

            import_end = time.perf_counter()
            import_time = import_end - import_start

            par_obj = normalise_objs(args, objs,  rot_angle=rotation_angle, offset_x=offset_x, offset_y=offset_y)

            # if object is spurious, there will be an exception
            func_start = time.perf_counter()
            render_views(args, out_dir, metadata_)
            func_end = time.perf_counter()
            render_time = func_end - func_start
        except Exception as e:
            log.error(f"Error processing {uid}: {e}")

        delete_objs(objs)

    total_processed = 0
    #Step 3: Render Rotation
    for model_path,uid in zip(full_paths, selected_uids):
        if do_exit:
            log.warning('process will restart...')
            break

        #Small hack. Sample floor and hdri ids now itself
        random_hdri_index = np.random.choice(range(0,len(hdri_list)))
        random_hdri_angle = np.random.uniform(0, 2 * np.pi)
        scene_type, _ = sample_hdri(args, hdri_list, index=random_hdri_index, angle=random_hdri_angle)
        floor_texture_index = np.random.choice(range(0,len(floor_textures[scene_type])))
       
        out_dir_rotation = Path(args.output_dir) / Path("rotation") / uid 
        if not args.reprocess and is_processed(out_dir_rotation, NUM_PER_OBJECT_ROTATION_RENDERS * args.num_render):
            log.warning(f'{uid} is already processed with {NUM_PER_OBJECT_ROTATION_RENDERS * args.num_render} renderings. Skipping...')
        else:
            for rotation_angle in np.arange(-np.pi/2,(np.pi/2)+1e-3,np.pi/(NUM_PER_OBJECT_ROTATION_RENDERS-1) ):
                log.info(f"Processing {uid}, Rotation : {(rotation_angle * 180./np.pi):.3f}")
                render_an_object(hdri_index=random_hdri_index, hdri_angle=random_hdri_angle,floor_index=floor_texture_index, out_dir=out_dir_rotation, model_path=model_path, uid=uid, rotation_angle=rotation_angle, offset_x=0.0, offset_y=0.0)

        out_dir_trans = Path(args.output_dir) / Path("translation") / uid 
        if not args.reprocess and is_processed(out_dir_trans, NUM_PER_OBJECT_TRANSLATION_RENDERS * args.num_render):
                log.warning(f'{uid} is already processed with {NUM_PER_OBJECT_TRANSLATION_RENDERS * args.num_render} renderings. Skipping...')
                #continue
        for translation in np.arange(-0.3,0.31,0.6/ (NUM_PER_OBJECT_TRANSLATION_RENDERS-1) ):
            log.info(f"Processing {uid}, Translation : {translation:.3f}")
            render_an_object(hdri_index=random_hdri_index, hdri_angle=random_hdri_angle,floor_index=floor_texture_index, out_dir=out_dir_trans, model_path=model_path, uid=uid, rotation_angle=0.0, offset_x=0.0, offset_y=translation)
        total_processed += 1
