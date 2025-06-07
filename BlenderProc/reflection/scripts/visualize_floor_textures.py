import blenderproc as bproc
import argparse
import bpy

from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes, Entity
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.material.MaterialLoaderUtility import (
    create_material_from_texture,
)
from blenderproc.python.material.MaterialLoaderUtility import create as create_material
import os
import time
from loguru import logger as log
from pathlib import Path
from mathutils import Vector, Matrix
import numpy as np
import math
import re
from typing import List, Optional, Dict
import json
import datetime
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()


def load_obj(
    filepath: str,
    cached_objects: Optional[Dict[str, List[MeshObject]]] = None,
    use_legacy_obj_import: bool = False,
    **kwargs,
) -> List[MeshObject]:
    """Import all objects for the given file and returns the loaded objects

    In .obj files a list of objects can be saved in.
    In .ply files only one object can be saved so the list has always at most one element

    :param filepath: the filepath to the location where the data is stored
    :param cached_objects: a dict of filepath to objects, which have been loaded before, to avoid reloading
                           (the dict is updated in this function)
    :param use_legacy_obj_import: If this is true the old legacy obj importer in python is used. It is slower, but
                                  it correctly imports the textures in the ShapeNet dataset.
    :param kwargs: all other params are handed directly to the bpy loading fct. check the corresponding documentation
    :return: The list of loaded mesh objects.
    """

    def filter_objs():
        """for importing only required objects to the scene"""
        if not kwargs.get("mirror", False):
            return
        valid_imports = ["mirror", "frame", "floor"]
        valid_types = ["MESH"]
        to_delete_objs = []
        for obj in bpy.context.selected_objects:
            if (
                obj.name not in valid_imports
                or obj.type not in valid_types
            ):
                to_delete_objs.append(obj)
        with bpy.context.temp_override(selected_objects=to_delete_objs):
            bpy.ops.object.delete()


    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The given filepath does not exist: {filepath}")

    if cached_objects is not None and isinstance(cached_objects, dict):
        if filepath in cached_objects.keys():
            created_obj = []
            for obj in cached_objects[filepath]:
                # duplicate the object
                created_obj.append(obj.duplicate())
            return created_obj
        loaded_objects = load_obj(filepath, cached_objects=None, **kwargs)
        cached_objects[filepath] = loaded_objects
        return loaded_objects
    # save all selected objects
    previously_selected_objects = bpy.context.selected_objects
    if filepath.endswith(".obj"):
        # load an .obj file:
        if use_legacy_obj_import:
            bpy.ops.import_scene.obj(filepath=filepath)  #, **kwargs)
        else:
            bpy.ops.wm.obj_import(filepath=filepath) #, **kwargs)
    elif filepath.endswith(".ply"):
        PLY_TEXTURE_FILE_COMMENT = "comment TextureFile "
        model_name = os.path.basename(filepath)

        # Read file
        with open(filepath, "r", encoding="latin-1") as file:
            ply_file_content = file.read()

        # Check if texture file is given
        if PLY_TEXTURE_FILE_COMMENT in ply_file_content:
            # Find name of texture file
            texture_file_name = re.search(
                f"{PLY_TEXTURE_FILE_COMMENT}(.*)\n", ply_file_content
            ).group(1)

            # Determine full texture file path
            texture_file_path = os.path.join(
                os.path.dirname(filepath), texture_file_name
            )
            material = create_material_from_texture(
                texture_file_path, material_name=f"ply_{model_name}_texture_model"
            )

            # Change content of ply file to work with blender ply importer
            new_ply_file_content = ply_file_content
            new_ply_file_content = new_ply_file_content.replace(
                "property float texture_u", "property float s"
            )
            new_ply_file_content = new_ply_file_content.replace(
                "property float texture_v", "property float t"
            )

            # Create temporary .ply file
            tmp_ply_file = os.path.join(Utility.get_temporary_directory(), model_name)
            with open(tmp_ply_file, "w", encoding="latin-1") as file:
                file.write(new_ply_file_content)

            # Load .ply mesh
            bpy.ops.import_mesh.ply(filepath=tmp_ply_file, **kwargs)

        else:  # If no texture was given
            # load a .ply mesh
            bpy.ops.import_mesh.ply(filepath=filepath, **kwargs)
            # Create default material
            material = create_material("ply_material")
            material.map_vertex_color()
        selected_objects = [
            obj
            for obj in bpy.context.selected_objects
            if obj not in previously_selected_objects
        ]
        for obj in selected_objects:
            obj.data.materials.append(material.blender_obj)
    elif filepath.endswith(".dae"):
        bpy.ops.wm.collada_import(filepath=filepath)
    elif filepath.lower().endswith(".stl"):
        # load a .stl file
        bpy.ops.wm.stl_import(filepath=filepath, **kwargs)
        # add a default material to stl file
        mat = bpy.data.materials.new(name="stl_material")
        mat.use_nodes = True
        selected_objects = [
            obj
            for obj in bpy.context.selected_objects
            if obj not in previously_selected_objects
        ]
        for obj in selected_objects:
            obj.data.materials.append(mat)
    elif filepath.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif filepath.lower().endswith(".glb") or filepath.lower().endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath, merge_vertices=True)
    elif (
        filepath.lower().endswith(".usda")
        or filepath.lower().endswith(".usd")
        or filepath.lower().endswith(".usdc")
    ):
        bpy.ops.wm.usd_import(filepath=filepath)

    # TODO: decouple the mirror and the floor
    filter_objs()

    mesh_objects = convert_to_meshes(
        [
            obj
            for obj in bpy.context.selected_objects
            if obj not in previously_selected_objects and obj.type == "MESH"
        ]
    )

    # Add properties to all objects of the imported mesh
    for j, obj in enumerate(mesh_objects):
        obj.set_cp("model_path", filepath)
        if kwargs.get("mirror", False) and "mirror" in obj.get_name().lower():
            obj.set_cp("category_id", 1)
        else:
            obj.set_cp("category_id", kwargs.get("global_category_id", 0))

    if kwargs.get("merge_objects", False):
        # Merge all objects into one
        mesh_objects = [
            bproc.object.merge_objects(mesh_objects, kwargs.get("obj_name", "exemplar"))
        ]

    return mesh_objects


def find_root_obj(obj):
    par = obj.parent
    if not par:
        return obj
    else:
        return find_root_obj(par)


def obj_meshes():
    for obj in bpy.context.selected_objects:
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def objs_bbox():
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in obj_meshes():
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalise_objs(objs: List[MeshObject]):
    """Normalise the objects to have a scale of 1 and be centered at the origin.

    :param objs: The objects to normalise
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select()
    bbox_min, bbox_max = objs_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    par_obj = find_root_obj(bpy.context.selected_objects[0])
    par_obj.scale = par_obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = objs_bbox()
    offset = - (bbox_min + bbox_max) / 2
    par_obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    return Entity(par_obj)


def create_light(name, light_type, energy, location, rotation, radius=0.25):
    light = bproc.types.Light(light_type=light_type, name=name)
    light.set_energy(energy)
    light.set_location(location)
    light.set_scale((1, 1, 1))
    light.set_rotation_euler(rotation)
    light.set_radius(radius)
    return light


def three_point_lighting():
    # # Key light
    # key_light = create_light(
    #     name="KeyLight",
    #     light_type="AREA",
    #     energy=600,
    #     location=(4, 4, 5),
    #     rotation=(math.radians(-45), 0, math.radians(-45)),
    # )
    # key_light.blender_obj.data.size = 2

    # # Fill light
    # fill_light = create_light(
    #     name="FillLight",
    #     light_type="AREA",
    #     energy=600,
    #     location=(-4, 4, 5),
    #     rotation=(math.radians(-45), 0, math.radians(45)),
    # )
    # fill_light.blender_obj.data.size = 2

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


def sample_hdri(args, hdri_list):
    hdri_path = os.path.join(args.hdri, np.random.choice(hdri_list))
    rot = (0, 0, np.random.uniform(0, 2 * np.pi))
    bproc.world.set_world_background_hdr_img(hdri_path, rotation_euler=rot)


def init(args):
    """Initialise the blenderproc settings and optionally override 
    from the ones in DefaultConfig.py.
    """
    bproc.init()
    # TODO: vary and check the renderings and their quality
    bproc.renderer.set_light_bounces(
        diffuse_bounces=5, 
        glossy_bounces=5, 
        max_bounces=5
    )

    # activate depth and normal map rendering
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.enable_normals_output()


def delete_objs(objs: List[MeshObject]):
    if objs is not None:
        for obj in objs:
            obj.delete()


def render_views(out_dir: str):
    bproc.renderer.enable_segmentation_output(default_values={"category_id": 0})
    data = bproc.renderer.render()
    data = save_cam_states(data)
    bproc.writer.write_hdf5(out_dir, data)

def load_floor_textures(texture_path):
    floor_textures = bproc.loader.load_ccmaterials(texture_path)
    return floor_textures

def bulk_process(args, mirror: List[MeshObject], cam_poses: List[np.ndarray] = []):
   
    total_processed = 0
    
    hdri_list = os.listdir(args.hdri)
    floor_textures = load_floor_textures(args.textures)
    bproc.camera.set_resolution(256, 256)

    objs = load_obj(str(args.model_3d_path), use_legacy_obj_import=True, global_category_id=2)
    par_obj = normalise_objs(objs)
    for ind, tx_floor in enumerate(floor_textures):
        print(f"Processing Texture : {tx_floor.get_name()}")
        out_dir = Path(args.output_dir) / tx_floor.get_name()
        bproc.utility.reset_keyframes()
        sample_hdri(args, hdri_list)
        floor_obj = bproc.filter.one_by_attr(mirror, "name", "floor")
        floor_obj.replace_materials(tx_floor)

        cam_poses = sample_cams(cam_poses)
        render_views(out_dir)
        total_processed += 1

    log.info(f"Total Processed Files : {total_processed}")


def save_cam_states(data):
    # Collect state of the camera at all frames
    cam_states = []
    for frame in range(bproc.utility.num_frames()):
        cam_states.append(
            {
                "cam2world": bproc.camera.get_camera_pose(frame),
                "cam_K": bproc.camera.get_intrinsics_as_K_matrix(),
            }
        )
    # Adds states to the data dict
    data["cam_states"] = cam_states
    return data

def main(args):

    init(args)

    # load the mirror
    mirror = load_obj(args.mirror, mirror=True)

    three_point_lighting()

    # sample cam poses
    cam_poses = []
    if args.camera:
        # read the camera positions file and convert into homogeneous camera-world transformation
        with open(args.camera, "r") as f:
            for line in f.readlines():
                line = [float(x) for x in line.split(',')]
                position, euler_rotation = line[:3], line[3:6]
                matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
                cam_poses.append(matrix_world)

    cam_poses = [cam_poses[0]] # Only render 1st view for fast testing. We'll only use one camera pose for this script.

    bulk_process(args, mirror, cam_poses)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script creates renderings for different floor textures.")
    parser.add_argument(
        "--model_3d_path",
        nargs="?",
        required=True,
        help="Path to the single model for which the rendering needs to be generated",
    )
    parser.add_argument(
        "--camera",
        nargs="?",
        help="Path to the camera file, should be reflection/resources/cam_poses.txt",
        default="reflection/resources/cam_poses.txt",
    )
    parser.add_argument(
        "--mirror",
        help="Path to the scene.blend mirror file, should be reflection/resources/mirror.fbx",
        nargs="?",
        default="reflection/resources/mirror.fbx",
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
        "--output_dir",
        nargs="?",
        required=True,
        help="Path to where the final files, will be saved. could be reflection/output",
    )
    
    args = parser.parse_args()
    main(args)
