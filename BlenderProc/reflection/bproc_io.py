"""
bproc_io.py

Handles all input/output and file-related operations in the pipeline.

Functions:
- load_3d_obj: Loads and configures a 3D object into the BlenderProc scene.
- load_scene: Sets up the base scene.
- sample_hdri, sample_floor: Samples and applies HDRIs and floor textures.
- load_floor_textures: Loads floor texture assets.
- delete_objs, remove_selected_object: Removes objects from the scene.
- write_json_file, save_cam_states: Handles output data saving (camera poses, metadata).
- get_timestamp: Returns a timestamp string for filenames.
- is_processed: Checks if a scene was already processed.
- add_properties_to_imported_mesh: Adds custom category metadata to imported meshes.
"""


import os
import json
import re
import bpy
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict
import blenderproc as bproc
from loguru import logger as log
import datetime

from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.material.MaterialLoaderUtility import (
    create_material_from_texture,
    create as create_material,
)

from reflection.errors import (
    SpuriousObjException,
    check_object,
)

from reflection.geometry import Mirrors

def get_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

def write_json_file(data_, path_):
    with open(path_, 'w') as f:
        json.dump(data_, f, indent=4)

def remove_selected_object():
    for obj in bpy.context.selected_objects:
        # Unlink the object from the current collection
        bpy.data.objects.remove(obj, do_unlink=True)

def load_obj(filepath: str, 
             use_legacy_obj_import: bool = False,
             **kwargs):
    """Import all objects for the given file. This function is called by load_scene and load_3D_obj

    In .obj files a list of objects can be saved in.
    In .ply files only one object can be saved so the list has always at most one element

    :param filepath: the filepath to the location where the data is stored
    :param use_legacy_obj_import: If this is true the old legacy obj importer in python is used. It is slower, but
                                  it correctly imports the textures in the ShapeNet dataset.
    :param kwargs: all other params are handed directly to the bpy loading fct. check the corresponding documentation
    :return: The list of loaded mesh objects.
    """
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

def add_properties_to_imported_mesh(filepath, mesh_objects, **kwargs):
    # Add properties to all objects of the imported mesh
    for j, obj in enumerate(mesh_objects):
        obj.set_cp("model_path", filepath)
        if kwargs.get("mirror", False) and obj.get_name().startswith("mirror_"):
            obj.set_cp("category_id", 1)
        else:
            obj.set_cp("category_id", kwargs.get("global_category_id", 0))

    if kwargs.get("merge_objects", False):
        # Merge all objects into one
        mesh_objects = [
            bproc.object.merge_objects(mesh_objects, kwargs.get("obj_name", "exemplar"))
        ]
    
    return mesh_objects

def load_3d_obj(
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
        loaded_objects = load_3d_obj(filepath, cached_objects=None, **kwargs)
        cached_objects[filepath] = loaded_objects
        return loaded_objects
    # save all selected objects
    previously_selected_objects = bpy.context.selected_objects
    
    #Load obj API. This will load the object from glb, ply, fbx, .. files
    load_obj(filepath=filepath, use_legacy_obj_import=use_legacy_obj_import, kwargs=kwargs)

    is_spurious = False
    if kwargs.get("check_spurious", False):
        # Following code is for finding spurious objects 
        # Write logic to check the objects
        for node in bpy.context.selected_objects:
            if check_object(node):
                is_spurious = True
                break

    if is_spurious:
        # First delete the object and then raise the exception
        bpy.ops.object.delete()  #Check if deletes all the objects
        # A node is spurious, raise Exception
        raise SpuriousObjException("Spurious Object Error", 400) 

    filter_objs()

    mesh_objects = convert_to_meshes(
        [
            obj
            for obj in bpy.context.selected_objects
            if obj not in previously_selected_objects and obj.type == "MESH"
        ]
    )

    mesh_objects = add_properties_to_imported_mesh(filepath=filepath, mesh_objects=mesh_objects, **kwargs)

    return mesh_objects

def load_scene(
    filepath: str,
    use_legacy_obj_import: bool = False,
    **kwargs,
) -> List[MeshObject]:
    """Import all objects for the scene and returns the loaded objects

    In .obj files a list of objects can be saved in.
    In .ply files only one object can be saved so the list has always at most one element

    :param filepath: the filepath to the location where the data is stored
    :param use_legacy_obj_import: If this is true the old legacy obj importer in python is used. It is slower, but
                                  it correctly imports the textures in the ShapeNet dataset.
    :param kwargs: all other params are handed directly to the bpy loading fct. check the corresponding documentation
    :return: The list of loaded mesh objects.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The given filepath does not exist: {filepath}")
    
    # save all selected objects
    previously_selected_objects = bpy.context.selected_objects
    
    #Load obj API. This will load the object from glb, ply, fbx, .. files
    load_obj(filepath=filepath, use_legacy_obj_import=use_legacy_obj_import, kwargs=kwargs)

    #Add mirrors data and Create map of mirros and corresponding objects. 
    mirrors_data = Mirrors()
    mesh_objects  = []
    for obj in bpy.context.selected_objects:
         if obj not in previously_selected_objects and obj.type == "MESH":
            mesh_objects.append(MeshObject(obj))
            if obj.name.startswith("mirror_"):
                mirrors_data.add_mirror(obj.name, mesh_objects[-1])
            elif obj.name.startswith("frame_"):
                if obj.name.startswith("frame_vm"):
                    #combine vanity mirror
                    mirrors_data.add_frame("frame_vm", mesh_objects[-1])
                else:
                    mirrors_data.add_frame(obj.name, mesh_objects[-1])

    mesh_objects = add_properties_to_imported_mesh(filepath=filepath, mesh_objects=mesh_objects, **kwargs)

    return mesh_objects, mirrors_data

def is_processed(dir: Path, num: int = 3):
    """checks if this object is already processed with the num of renderings"""
    return len(list(dir.glob("*.hdf5"))) == num

def sample_hdri(args, hdri_list, index=None, angle=None):
    if index is not None:
        hdri_path = hdri_list[index]
    else:
        hdri_path = np.random.choice(hdri_list)
    if angle is not None:
        rot = (0, 0, angle)
    else:
        rot = (0, 0, np.random.uniform(0, 2 * np.pi))
    bproc.world.set_world_background_hdr_img(hdri_path, rotation_euler=rot)
    scene_type = "outdoor" if "outdoor" in hdri_path else "indoor"
    return scene_type, os.path.join( scene_type, os.path.split(hdri_path)[1] )

def delete_objs(objs: List[MeshObject]):
    if objs is not None:
        for obj in objs:
            obj.delete()

#####################################################################################
#                                                                                   #
# Floor texture Code Starts                                                         #
#####################################################################################


def load_floor_textures(texture_path:str)->Dict:
    """
    Loads floor textures from the specified file path.

    Args:
        texture_path (str): The file path to the directory containing the floor texture files.

    Returns:
        Dict: A dictionary mapping indoor and outdoor textures to their corresponding data.
    """
    indoor_floor_textures = bproc.loader.load_ccmaterials(os.path.join(texture_path,'Indoor'))
    outdoor_floor_textures = bproc.loader.load_ccmaterials(os.path.join(texture_path,'Outdoor'))
    log.info(f"No. of Indoor Textures: {len(indoor_floor_textures)}\t No. of Outdoor Textures: {len(outdoor_floor_textures)}")
    return {"indoor":indoor_floor_textures, "outdoor":outdoor_floor_textures}

def sample_floor(floor_obj, floor_texture_list, index=None):
    """
    Samples a random floor texture from the provided list and applies it to the given floor object.

    Args:
        floor_obj (Object): The floor object to which the texture will be applied.
        floor_texture_list (List): A list of floor texture data, typically obtained from load_floor_textures().

    Returns:
        None
    """
    if index is not None:
        random_floor_texture = floor_texture_list[index]
    else:
        random_floor_texture = np.random.choice(floor_texture_list)
    floor_obj.replace_materials(random_floor_texture)
    return random_floor_texture.get_name()

#####################################################################################
#                                                                                   #
# Floor texture Code Ends                                                           #
#####################################################################################

def save_cam_states(data, metadata):
    # Collect state of the camera at all frames
    cam_states = []
    for frame in range(bproc.utility.num_frames()):
        cam_states.append(
            {
                "cam2world": bproc.camera.get_camera_pose(frame),
                "cam_K": bproc.camera.get_intrinsics_as_K_matrix(),
                "metadata" : metadata
            }
        )
    # Adds states to the data dict
    data["cam_states"] = cam_states
    return data

