"""
errors.py

Handles validation, error checking, and logging for spurious or invalid objects.

Functions:
- check_object: Master validation function for object geometry, material, and nodes.
- check_mesh_properties: Checks mesh data integrity.
- check_material_properties, check_node_properties: Validate Blender material properties.
- is_spurious_object: Heuristically detects unwanted/spurious 3D models.
- count_spurious_files: Logs statistics on failed or unusable objects.

Classes:
- SpuriousObjException, CollisionException: Custom exceptions for handling invalid object states.
- ErrorRecord: Tracks and logs errors across scenes and objects.
"""


import bpy
import json
from loguru import logger as log


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

class CollisionException(Exception):
    """Custom exception for collision cases
    """
    def __init__(self, message):
        super().__init__(f"CollisionException: {message}")

def count_spurious_files(data_):
    total = 0
    for key in data_.keys():
        total += len(data_[key])
    return total

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
