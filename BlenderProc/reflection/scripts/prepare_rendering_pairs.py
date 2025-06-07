import blenderproc as bproc
import bpy
import os
import re

import random
import json

from blenderproc.python.types.MeshObjectUtility import MeshObject, convert_to_meshes, Entity
from blenderproc.python.utility.Utility import Utility
from blenderproc.python.material.MaterialLoaderUtility import (
    create_material_from_texture,
)
from blenderproc.python.material.MaterialLoaderUtility import create as create_material
from typing import List, Optional, Dict

import math
import time
from mathutils import Vector, Matrix

#Code is repeated here from main/reflection.py
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

def load_abo_paired_data():
    with open("reflection/resources/abo_multi_obj.json",'r') as f:
        abo_multiple_obj = json.load(f)
    return abo_multiple_obj

def delete_objs(objs: List[MeshObject]):
    if objs is not None:
        for obj in objs:
            obj.delete()

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

    obj_bbox = objs_bbox()

    #import pdb; pdb.set_trace()
    bpy.ops.object.select_all(action="DESELECT")
    return Entity(par_obj), obj_bbox

def are_sizes_similar(first_box, second_box):
    def get_dimensions(min_box, max_box):
        return max_box[0] - min_box[0], max_box[1] - min_box[1], max_box[2] - min_box[2]
    
    f_dim_1 = get_dimensions(first_box[0], first_box[1])
    s_dim_2= get_dimensions(second_box[0], second_box[1])

    def are_close(dim_a, dim_b):
        """Check if ratio of sides is > 0.8"""
        if dim_a < dim_b:
            return True if (dim_a/dim_b) >= 0.8 else False
        else:
            return True if (dim_b/dim_a) >= 0.8 else False

    count = 0
    if not are_close( f_dim_1[0],  s_dim_2[0]):
        return False

    if not are_close( f_dim_1[1],  s_dim_2[1]):
        return False

    if not are_close( f_dim_1[1],  s_dim_2[1]):
       return False

    
    return True

def save_paired_data(paired_data):
    with open("reflection/resources/abo_paired.json",'w') as f:
        json.dump(paired_data, f, indent=4)

mul_obj_paired_data = load_abo_paired_data()

# Shared dictionary to store results
results = {}

#Create a file with paired_data and add random_seed
with open("reflection/resources/abo_paired.json",'r') as f:
    paired_data = json.load(f)

random.seed(paired_data['random_seed'])

print(f"Loaded pre-loaded data : {len(paired_data.keys()) - 1}")

iters = 0
count = 0
for uid, path in mul_obj_paired_data['path_data'].items():
    iters += 1
    if count > 1000:
        break
    print(f"Processing {iters} : {uid}")
    if uid in paired_data.keys():
        if paired_data[uid] is not None:
            continue
    count += 1
    import_start = time.perf_counter()
    first_obj = load_3d_obj(path, use_legacy_obj_import=True, global_category_id=2, check_spurious=False)
    par_obj, par_obj_bbox = normalise_objs(first_obj)
    import_end = time.perf_counter()
    import_time = import_end - import_start

    print(f"{uid} First obj loaded in {import_time} seconds")

    first_obj_token = uid
                        
    #Step 1. Find categroy id for this object
    first_obj_category = mul_obj_paired_data["glb_data"][first_obj_token]

    #Step 2. Find the cluster id for this category
    cluster_id_to_sample_from = mul_obj_paired_data["category_to_cluster"][first_obj_category]

    max_tries = 0
    while True:

        #Step 4. Sample another category from the cluster
        sampled_category = random.choice( mul_obj_paired_data["paired_data"][cluster_id_to_sample_from] ) 

        #Step 5. Sample object from this cluster
        second_obj_fname = random.choice( mul_obj_paired_data["category_data"][sampled_category] )

        second_objs = load_3d_obj(mul_obj_paired_data["path_data"][second_obj_fname], use_legacy_obj_import=True, global_category_id=3, check_spurious=False)
        par_second_obj, second_obj_bbox = normalise_objs(second_objs)

        are_boxes_of_same_size = are_sizes_similar(par_obj_bbox, second_obj_bbox)

        delete_objs(second_objs)
        max_tries += 1
        if are_boxes_of_same_size:
            paired_data[uid] = second_obj_fname
            break
        if max_tries > 20:
            print(f"{uid} : Reched maximum trials")
            paired_data[uid]  = None
            break

    delete_objs(first_obj)
    if iters % 20 == 0:
        save_paired_data(paired_data)

    ###paired_data[uid] =  

save_paired_data(paired_data)