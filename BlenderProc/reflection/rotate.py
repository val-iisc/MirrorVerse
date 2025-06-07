import blenderproc as bproc
import argparse
import bpy
import random

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
import glob
# import debugpy
# debugpy.listen(5678)
# debugpy.wait_for_client()
# debugpy.breakpoint()

### MAPPING LIST
MIRROR_FRAME_MAPPING = {
    "frame_1" : "mirror_common",
    "frame_2" : "mirror_common",
    "frame_3" : "mirror_common",
    "frame_4" : "mirror_common",
    "frame_5" : "mirror_common",
    "frame_6" : "mirror_common",
    "frame_vm" : "mirror_vm_straight",
    "frame_base" : "mirror_base"
}
def get_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return timestamp

"""
The following code (for finding spurioous objects) is copied from scripts/find_spurious.py
"""
def is_spurious_object(node, node_input, linked_node):
    if node.name=="Mix Shader" and node_input.name=="Fac" and linked_node.name=="Light Path":
        return True
    return False

def check_node_properties(node, indent=""):
    #print(f"{indent}Node: {node.name}, Type: {node.type}")
    is_spurious = False
    for node_input in node.inputs:
        if node_input.is_linked:
            linked_node = node_input.links[0].from_node
            linked_socket = node_input.links[0].from_socket
            #print(f"{indent}  Input: {node_input.name}, Linked to {linked_node.name} - {linked_socket.name}")
            if is_spurious_object(node, node_input, linked_node):
                is_spurious = True
                break
        # else:
        #     if hasattr(node_input, 'default_value'):
        #         #print(f"{indent}  Input: {node_input.name}, Value: {node_input.default_value}")
        #     else:
        #         #print(f"{indent}  Input: {node_input.name}, No value")

    return is_spurious

        

def check_material_properties(material, indent=""):
    #print(f"{indent}Material: {material.name}")
    is_spurious = False
    if material.use_nodes:
        node_tree = material.node_tree
        for node in node_tree.nodes:
            if check_node_properties(node, indent + "  "):
                is_spurious = True
                break

    return is_spurious

def check_mesh_properties(obj, indent=""):
    mesh = obj.data
    is_spurious = False
    for mat_slot in obj.material_slots:
        material = mat_slot.material
        if material:
            if check_material_properties(material, indent + "  "):
                is_spurious = True
                break
    return is_spurious

def check_object(obj, indent=""):
    is_spurious = False
    if obj.type == 'MESH':
        is_spurious = check_mesh_properties(obj, indent + "  ")

    if is_spurious:
        return True

    for child in obj.children:
        is_spurious = check_object(child, indent + "  ")
        if is_spurious:
            return True

    return False

class SpuriousObjException(Exception):
    """Custom exception with additional attributes."""
    """error code:
        400 - MixedShader is of type Fac
    """
    def __init__(self, message, error_code):
        super().__init__(message)
        self.error_code = error_code

def write_json_file(data_, path_):
    with open(path_, 'w') as f:
        json.dump(data_, f, indent=4)

def count_spurious_files(data_):
    total = 0
    for key in data_.keys():
        total += len(data_[key])
    return total

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

class Mirrors:
    def __init__(self):
        self.frames = {}
        self.mirrors = {}

    def add_frame(self, id, bproc_obj):
        if id not in self.frames.keys():
            self.frames[id] = [] 
        self.frames[id].append(bproc_obj)

    def add_mirror(self, mirror_id, bproc_obj):
        self.mirrors[mirror_id] = bproc_obj

    def __len__(self):
        return len(self.frames)

    def hide_all_mirrors(self):
        for _, frames_list in self.frames.items():
            for fm in frames_list:
                if not fm.is_hidden():
                    fm.hide()

        for _, mirrors in self.mirrors.items():
            if not mirrors.is_hidden():
                mirrors.hide()

    def select_mirror_randomly(self):
        small_frame_keys = [key for key in self.frames.keys() if key != "frame_base"]
        random_frame = random.choice(small_frame_keys)
        for frame_obj in self.frames[random_frame]:
            frame_obj.hide(hide_object=False) 
        #Also make corresponding mirror visible
        self.mirrors[ MIRROR_FRAME_MAPPING[random_frame] ].hide(hide_object=False)


    def select_base_mirror(self):
        base_mirror_id = "frame_base"
        for frame_obj in self.frames[base_mirror_id]:
            frame_obj.hide(hide_object=False)  
        self.mirrors[ MIRROR_FRAME_MAPPING[base_mirror_id] ].hide(hide_object=False) 

    def select_mirror(self, only_base_mirror=True):
        if only_base_mirror:
            self.select_base_mirror()
        else:
            self.select_mirror_randomly()

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
    hdri_path = np.random.choice(hdri_list)
    rot = (0, 0, np.random.uniform(0, 2 * np.pi))
    bproc.world.set_world_background_hdr_img(hdri_path, rotation_euler=rot)
    scene_type = "outdoor" if "outdoor" in hdri_path else "indoor"
    return scene_type, os.path.join( scene_type, os.path.split(hdri_path)[1] )

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

    #enable_depth_output function can only be called once. Please be careful!!
    bproc.renderer.enable_depth_output(activate_antialiasing=False)

def delete_objs(objs: List[MeshObject]):
    if objs is not None:
        for obj in objs:
            obj.delete()


def render_views(out_dir: str, metadata : Dict):
    # activate normal map rendering. This needs to be called after add_camera_pose !!
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_segmentation_output(default_values={"category_id": 0})
    data = bproc.renderer.render()
    data = save_cam_states(data, metadata)
    bproc.writer.write_hdf5(out_dir, data)


def is_processed(dir: Path, num: int = 3):
    """checks if this object is already processed with the num of renderings"""
    return len(list(dir.glob("*.hdf5"))) == num

class ErrorRecord:
    def __init__(self, code_id=300):
        self.error_codes = {}
        self.error_counter = code_id #Error code will start from 300

    def generate_new_key(self, message):
        e_code = self.error_counter
        self.error_counter += 1
        self.error_codes[message] = e_code
        return e_code
        
    def get_error_code(self, message):
        #Check if this message is previously encountered:
        if message in self.error_codes.keys():
            return self.error_codes[message]
        else:
            return self.generate_new_key(message)

    def write_generic_error(self, spurious_data, message, file_uid):
        e_code = self.get_error_code(message)
        if e_code not in spurious_data.keys():
            spurious_data[e_code] = []
        spurious_data[e_code].append(file_uid)       

    def get_error_codes(self):
        return self.error_codes

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

def sample_floor(floor_obj, floor_texture_list):
    """
    Samples a random floor texture from the provided list and applies it to the given floor object.

    Args:
        floor_obj (Object): The floor object to which the texture will be applied.
        floor_texture_list (List): A list of floor texture data, typically obtained from load_floor_textures().

    Returns:
        None
    """
    random_floor_texture = np.random.choice(floor_texture_list)
    floor_obj.replace_materials(random_floor_texture)
    return random_floor_texture.get_name()

#####################################################################################
#                                                                                   #
# Floor texture Code Ends                                                           #
#####################################################################################

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

            scene_type, hdri_info = sample_hdri(args, hdri_list)
            
            # Sample Floor texture
            floor_obj = bproc.filter.one_by_attr(mirror, "name", "floor")
            floor_texture_info = sample_floor(floor_obj, floor_textures[scene_type] ) 
            
            # Metadata
            metadata_ = {'hdri' : hdri_info, 'floor_texture' : floor_texture_info}

            log.info(f"Processing {file.stem}")
            objs = None
            try:
                cam_poses = sample_cams(cam_poses)
                if args.small_mirrors:
                    diff_mirrors_data.select_mirror(only_base_mirror=False)
                import_start = time.perf_counter()

                step = - np.pi/4
                for index,angle in enumerate(np.arange(step, (np.pi/4) + 0.1, np.pi/36)):
                    objs = load_3d_obj(str(file), use_legacy_obj_import=True, global_category_id=2, check_spurious=args.check_spurious)

                    import_end = time.perf_counter()
                    import_time = import_end - import_start

                    out_dir_local = Path(f"{str(out_dir )}_angle_{index}")
                    # TODO fix this. log to spurious file with different error code and check.
                    # if import_time > 60:
                    #     log.warning(f"Importing {file.stem} took {import_time} seconds. Skipping large import files.")
                    #     delete_objs(objs)
                    #     continue

                    par_obj = rotate_obj(objs, angle)

                    # if object is spurious, there will be an exception
                    func_start = time.perf_counter()
                    render_views(out_dir_local, metadata_)
                    func_end = time.perf_counter()
                    render_time = func_end - func_start
                
                    if import_time < 10 and render_time > args.max_render_time:
                        do_exit = True
                        break

                    delete_objs(objs)
            except SpuriousObjException as e:
                log.error(f"Error processing {file.stem}: {e}. Error Code: {e.error_code}")
                spurious_uids.add(file.stem)

            except Exception as e:
                log.error(f"Error processing {file.stem}: {e}")
                # Write spurious file uid in a dictionary
                generic_error_recoder.write_generic_error(spurious_files, f"{e}", file.stem)

            delete_objs(objs)
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
    if not do_exit:
        log.info("Process completed successfully.")

def rotate_obj(objs: List[MeshObject], angle):
    """Rotates the object around it's center along it's y-axis.

    :param objs: The objects to normalise
    :param angle: angle by which objects need to be rotated (in radians)
    """ 
    
    def get_transformation_matrix(centroid, scale):
        """Get the transformation matrix for the root node. 

        :param centroid: centroid of the bounding cuboid aroung the object
        :param scale: factor by which object need to be scaled. 
        """
        #First transformaion. shift origin to centroid
        shift_origin_to_centroid = np.eye(4, dtype=np.float32)
        shift_origin_to_centroid[0,3] = -centroid[0]
        shift_origin_to_centroid[1,3] = -centroid[1]
        shift_origin_to_centroid[2,3] = -centroid[2]

        #Rotate along y-axis by angle
        rotate_tfm = np.eye(4, dtype=np.float32)
        rotate_tfm[0,0] = math.cos(angle)
        rotate_tfm[2,2] = math.cos(angle)
        rotate_tfm[0,2] = -math.sin(angle)
        rotate_tfm[2,0] = math.sin(angle)

        ##Shift back to centroid
        shift_back = np.eye(4, dtype=np.float32)
        shift_back[0,3] = centroid[0]
        shift_back[1,3] = centroid[1]
        shift_back[2,3] = centroid[2]

        ##Scale matrix
        scale_mat = scale * np.eye(4, dtype=np.float32)

        #Remember transformation order is Scale, Rotation & Translation.
        return Matrix( shift_back  @ rotate_tfm  @ scale_mat @ shift_origin_to_centroid )

    #Step 1 : Select the object
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select()

    #Find bounding box of this object
    bbox_min, bbox_max = objs_bbox()
    scale = 1 / max(bbox_max - bbox_min)

    #Find centroid of this selection
    centroid = (bbox_max + bbox_min)/2.
    tfm_matrix = get_transformation_matrix(centroid, scale)

    par_obj = find_root_obj(bpy.context.selected_objects[0])
   
    par_obj.matrix_world  = par_obj.matrix_world @ tfm_matrix
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
   
    #Ground the object
    #Find bounding box of this object again
    bbox_min, bbox_max = objs_bbox()
    if bbox_min[2] > -0.5:
        curr_world_matrix = np.array(par_obj.matrix_world)
        curr_world_matrix[:3,3] = -0.5 - bbox_min[2]
        par_obj.matrix_world =  Matrix(curr_world_matrix)
        bpy.context.view_layer.update()

    bpy.ops.object.select_all(action="DESELECT")
    return Entity(par_obj)

def get_transformation_matrix(objs: List[MeshObject]):
    #Step 1 : Select the object
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objs:
        obj.select()
    par_obj = find_root_obj(bpy.context.selected_objects[0])
   
    return par_obj.matrix_world


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
    
    
def main(args):

    init(args)

    # load the mirror
    mirror, diff_mirrors_data = load_scene(args.mirror, mirror=True)

    # create lights
    # create_light(
    #     name="PointLight",
    #     light_type="POINT",
    #     energy=600,
    #     location=(0, -0.75, 1.0),
    #     radius=0,
    # )
    three_point_lighting()

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
    selected_indices = random_pose_indices(len(all_cam_poses), args.num_render)
    for index in selected_indices:
        cam_poses.append(all_cam_poses[index])    

    if args.fast_testing:
        cam_poses = [cam_poses[0]] # Only render 1st view for fast testing.

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
        "--fast_testing",
        action='store_true', default=False,
        help="Enable it so that changes can be tested quickly. Use degraded options",
    )
  
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Whether to reprocess. This does not check `is_processed` and can reprocess files. NOTE: Do not use this with `rerun.py`.",
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
    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)
        os.environ["BLENDER_PROC_RANDOM_SEED"] = str(args.seed)
    main(args)
