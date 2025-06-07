import blenderproc as bproc
import argparse
import glob
import os, sys
import bpy

"""
Identified properties for bhutiya objects:

1.) Node: Mix Shader, Type: MIX_SHADER
    Input: Fac, Linked to Light Path - Is Camera Ray

"""
def does_dir_exist(dir_path):
    if not os.path.exists(dir_path):
        print(f"Input path does not exist")
        sys.exit(-1)

    if not os.path.isdir(dir_path):
        print(f"Input path is not a directory")
        sys.exit(-1)

def remove_all_objects():
    # Remove all objects from the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def load_gltf(filepath):
    if not filepath.endswith(('.gltf', '.glb')):
        print("The file must be a .gltf or .glb file.")
        return

    bpy.ops.import_scene.gltf(filepath=filepath)
    print(f"Loaded GLTF file: {filepath}")

    # Get the last added object (the one that was just imported)
    imported_object = bpy.context.selected_objects[-1]

    # Set the imported object as the active object
    bpy.context.view_layer.objects.active = imported_object

    #print(f"Active object set to: {imported_object.name}")

def is_spurious_object(node, node_input, linked_node):
    #import pdb; pdb.set_trace()
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

def main(args):
    #Sanity check
    does_dir_exist(args.input_dir)

    #Find glb files in the directory

    glb_file_paths = glob.glob( os.path.join(args.input_dir, "**/*.glb"), recursive=True)

    #glb_file_paths = ['/home/test/ankit/6f99fb8c2f1a4252b986ed5a765e1db9.glb']

    spurious_files = []

    for f_path in glb_file_paths:
        load_gltf(f_path)
        active_object = bpy.context.active_object
        if active_object:
            if check_object(active_object):
                basedir, f_name = os.path.split(f_path)
                spurious_files.append(f"{os.path.join(os.path.split(basedir)[1],f_name)}")
        # else:
        #     print("No active object selected.")

        remove_all_objects()

    print(spurious_files)

    with open(args.output_file, 'w') as f:
        for sp_path in spurious_files:
            f.write(f"{sp_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find files which doe not form reflections because of some issues")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/manan/data/objaverse/blenderproc",
        required=True,
        help="Directory where 3d models are saved"
    )
    parser.add_argument(
        "--output_file", type=str, default="spurious_files.txt", help="Name of the output txt file"
    )
 
    args = parser.parse_args()
    main(args)