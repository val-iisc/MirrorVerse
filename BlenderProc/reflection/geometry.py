"""
geometry.py

Handles geometric operations on 3D objects, such as scaling, grounding, rotation, and bounding box calculation.
Also includes mirror and frame handling logic.

Functions:
- normalise_objs: Scales and centers 3D objects for consistent placement.
- ground_the_object: Moves object to rest on the floor plane.
- update_world_matrix_with_new_rotation, get_rotation_mat: Applies and computes rotation matrices.
- obj_meshes, objs_bbox: Gathers mesh data and calculates bounding boxes.
- find_root_obj: Finds the topmost parent of an object.
- Mirrors (class): Manages mirror-frame pairs and their configuration in the scene.
"""


import math
import numpy as np
import bpy
import mathutils
from mathutils import Vector, Matrix
import random
from typing import List

from blenderproc.python.types.MeshObjectUtility import MeshObject, Entity

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

# offsets to randomise position after normalisation to origin
RAND_POS_X_OFFSET = [-0.3, 0.3]
RAND_POS_Y_OFFSET = [-0.3, 0.3]


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

def get_rotation_mat(entity_obj, rot_angle=None):
    """
    Randomly rotates an object along it's z-axis and returns the rotation matrix
    of a given entity object. 
    Random angle is sampled between [-pi/2, pi/2]
    Parameters:
    -----------
    entity_obj : Input mesh.

    Returns:
    --------
    numpy.ndarray
        A 3x3 numpy array representing the rotation matrix of the input mesh
    """
    #Sample a random angle 
    if rot_angle is None:
        rotation_angle = np.random.uniform(-np.pi/2, np.pi/2)
    else:
        rotation_angle = rot_angle
    curr_rotation_euler = entity_obj.get_rotation_euler()
    curr_rotation_euler[2] += rotation_angle
    entity_obj.set_rotation_euler(curr_rotation_euler)
    return entity_obj.get_rotation_mat()

def update_world_matrix_with_new_rotation(world_matrix, rotation_tfm):
    """
    Updates the world matrix of an object with a new rotation transformation.

    This function modifies the given world matrix by applying a new rotation transformation.
    The world matrix typically represents the combined transformations (translation, rotation,
    and scaling) of an object in 3D space. By updating the rotation component, the function
    ensures that the object's orientation is adjusted accordingly.

    Parameters:
    -----------
    world_matrix : mathutils.Matrix
        The current 4x4 world matrix of the object. 

    rotation_tfm : numpy.ndarray
        A 3x3 rotation matrix representing the new rotation to be applied to the object.
        This matrix should be orthogonal and represent a valid rotation in 3D space.

    Returns:
    --------
     mathutils.Matrix
        The updated 4x4 world matrix after applying the new rotation transformation. The
        translation component of the world matrix remains unchanged, while the rotation
        component is updated to reflect the new orientation.
    """

    #Decompose the world-matrix to Translation, rotation and scale
    translation, rotation_quaternion, scale = world_matrix.decompose()

    #Convert Quaternion to rotation matrix
    curr_rotation_matrix = rotation_quaternion.to_matrix()

    #pre-multiply new rotation matrix
    new_rotation_np = rotation_tfm @ np.array(curr_rotation_matrix)

    # Create translation matrix
    translation_matrix = mathutils.Matrix.Translation(translation)

    # Create rotation matrix
    new_rotation_matrix = mathutils.Matrix(new_rotation_np.tolist()).to_4x4()

    # Create scale matrix
    scale_matrix = mathutils.Matrix.Diagonal(scale).to_4x4()

    # Combine matrices to create the new world matrix
    new_world_matrix = translation_matrix @ new_rotation_matrix @ scale_matrix

    return new_world_matrix

def ground_the_object(bbox_min, offset):
    """
    Adjusts the position of an object to ensure it is grounded based on its 
    bounding box and an offset. This function calculates the new position of
    an object such that its lowest point (based on the bounding box)
    is aligned with the ground plane.

    Parameters:
    -----------
    bbox_min : mathutils.Vector
        The minimum value of the bounding box of shape (3,).

    offset : mathutils.Vector
        The offset based on the centroid of the original bounding box.

    Returns:
    --------
    float 
        Additional offset to be added along the negative z-axis.
    """

    #Already negative sign is added to the centroid offset
    bbox_min_after_offset = bbox_min + offset

    #Check the z value of this new offset, floor is at -0.5
    floor_z_coordinate = -0.5
    vertical_offset_to_ground = 0.0
    if bbox_min_after_offset[2] > floor_z_coordinate: 
        vertical_offset_to_ground = bbox_min_after_offset[2] - floor_z_coordinate

    return vertical_offset_to_ground if vertical_offset_to_ground > 0.01 else 0.0

def normalise_objs(args, objs: List[MeshObject], rot_angle=None, offset_x=None, offset_y=None):
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

    if not args.disable_rotate:
        #Irrespective of the mirror-type, rotate the object along vertical axis
        #Let's make a dummy object, so that we can call for the sampler
        dummy_entity_obj = Entity(par_obj)
    
        if rot_angle is None:
            rotation_mat = get_rotation_mat(dummy_entity_obj)
        else:
            rotation_mat = get_rotation_mat(dummy_entity_obj, rot_angle=rot_angle)

        par_obj.matrix_world = update_world_matrix_with_new_rotation(par_obj.matrix_world, rotation_mat)

    ground_offset_z = ground_the_object(bbox_min, offset)
    if ground_offset_z > 0.0:
        #negative sign is added because we need to move along negative z-axis
        offset += Vector([0, 0, -ground_offset_z])

    if not args.small_mirrors:
        # Generate random offsets within the specified ranges
        if offset_x is not None and offset_y is not None:
            rand_x = offset_x
            rand_y = offset_y
        else:
            rand_x = random.uniform(RAND_POS_X_OFFSET[0], RAND_POS_X_OFFSET[1])
            rand_y = random.uniform(RAND_POS_Y_OFFSET[0], RAND_POS_Y_OFFSET[1])
        offset += Vector([rand_x, rand_y, 0])

    par_obj.matrix_world.translation += offset

    obj_bbox = objs_bbox()

    bpy.ops.object.select_all(action="DESELECT")
    return Entity(par_obj)
