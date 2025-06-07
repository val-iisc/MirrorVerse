import bpy
import math
import os
import numpy as np
import math
import argparse
import sys
from mathutils import Quaternion
import time
import mathutils


def cartesian_to_spherical(x, y, z):
    # Calculate radius (r)
    r = math.sqrt(x**2 + y**2 + z**2)

    # Calculate polar angle (θ)
    if r > 0:
        if z > 0:
            theta = math.atan2(math.sqrt(x**2 + y**2), z)
        elif z < 0:
            theta = math.pi + math.atan2(math.sqrt(x**2 + y**2), z)
        elif z == 0:
            theta = math.pi / 2
        else:
            theta = 0.0  # undefined case.
    else:
        theta = 0.0  # Handle division by zero case

    # Calculate azimuthal angle (φ)
    phi = math.atan2(y, x)

    if x > 0:
        phi = math.atan2(y, x)
    elif x < 0 and y >= 0:
        phi = math.atan2(y, x) + math.pi
    elif x < 0 and y < 0:
        phi = math.atan2(y, x) - math.pi
    elif x == 0 and y > 0:
        phi = math.pi / 2
    elif x == 0 and y < 0:
        phi = -math.pi / 2
    else:
        phi = 0

    # Convert angles to degrees if needed
    theta_degrees = math.degrees(theta)
    phi_degrees = math.degrees(phi)

    return r, theta_degrees, phi_degrees  # Return spherical coordinates (r, θ, φ)


def spherical_to_cartesian(r, theta, phi):
    # Convert angles from degrees to radians if needed
    theta_rad = math.radians(theta)
    phi_rad = math.radians(phi)

    # Calculate Cartesian coordinates
    x = r * math.sin(theta_rad) * math.cos(phi_rad)
    y = r * math.sin(theta_rad) * math.sin(phi_rad)
    z = r * math.cos(theta_rad)

    return x, y, z  # Return Cartesian coordinates (x, y, z)


def sample_camera_locations(mirror_location, camera_location):
    # Mirror location is origin
    rel_x = camera_location.x - mirror_location[0]
    rel_y = camera_location.y - mirror_location[1]
    rel_z = camera_location.z - mirror_location[2]

    # This position's r,theta, phi
    r, theta, phi = cartesian_to_spherical(rel_x, rel_y, rel_z)

    # now keeping radius constant, create new poses between [theta-30, theta + 30] & [phi - 30, phi + 30]
    new_camera_locations = []

    sampling_interval = 5

    for new_theta in np.arange(theta, theta + 0.1, 2):
        for new_phi in np.arange(phi - 5, phi + 5.5, sampling_interval):
            new_pos = spherical_to_cartesian(r, new_theta, new_phi)
            # print("normalized ", new_pos, rel_x, rel_y, rel_z)
            new_camera_locations.append(
                (
                    new_pos[0] + mirror_location[0],
                    new_pos[1] + mirror_location[1],
                    new_pos[2] + mirror_location[2],
                )
            )
            print(new_camera_locations[-1])
    return new_camera_locations


def shift_location(inp_location, shift_fac):
    inp_vec = np.array(list(inp_location), dtype=np.float32)
    magnitude = np.sqrt(
        inp_vec[0] * inp_vec[0] + inp_vec[1] * inp_vec[1] + inp_vec[2] * inp_vec[2]
    )
    unit_vec = inp_vec / magnitude

    new_vec = inp_vec + shift_fac * unit_vec
    return tuple(new_vec.tolist())


def calculate_dimensions(obj):
    # Ensure object is selected and active
    bpy.context.view_layer.objects.active = obj

    # Update the scene to ensure the object's data is valid
    bpy.context.view_layer.update()

    # Get the dimensions of the object's bounding box
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_coords = bbox[0]
    max_coords = bbox[0]

    for coord in bbox:
        min_coords = min(min_coords, coord)
        max_coords = max(max_coords, coord)

    # Calculate dimensions (width, height, depth)
    dimensions = (
        max_coords[0] - min_coords[0],
        max_coords[1] - min_coords[1],
        max_coords[2] - min_coords[2],
    )

    return dimensions


def get_dimensions_for_hierarchical_mesh(gltf_obj):
    if gltf_obj:
        # Update the object's bounding box to ensure dimensions are calculated
        bpy.context.view_layer.update()

        # Function to recursively calculate combined dimensions of all mesh objects in the hierarchy
        def calculate_combined_dimensions(obj):
            # Initialize variables to track combined dimensions
            bbox_min = mathutils.Vector((float("inf"), float("inf"), float("inf")))
            bbox_max = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))

            # Traverse through all children recursively
            def traverse(obj):
                nonlocal bbox_min, bbox_max

                # Check if the object is a mesh with valid mesh data
                if obj.type == "MESH" and obj.data:
                    # Get local bounding box dimensions
                    mesh_bbox_min, mesh_bbox_max = obj.bound_box[0], obj.bound_box[6]

                    # Apply object's local transformation
                    # import pdb; pdb.set_trace()
                    mesh_bbox_min = obj.matrix_world @ mathutils.Vector(mesh_bbox_min)
                    mesh_bbox_max = obj.matrix_world @ mathutils.Vector(mesh_bbox_max)

                    # Update combined dimensions
                    bbox_min = mathutils.Vector(
                        (
                            min(bbox_min.x, mesh_bbox_min.x),
                            min(bbox_min.y, mesh_bbox_min.y),
                            min(bbox_min.z, mesh_bbox_min.z),
                        )
                    )
                    bbox_max = mathutils.Vector(
                        (
                            max(bbox_max.x, mesh_bbox_max.x),
                            max(bbox_max.y, mesh_bbox_max.y),
                            max(bbox_max.z, mesh_bbox_max.z),
                        )
                    )

                # Recursively traverse children
                for child in obj.children:
                    traverse(child)

            # Start traversal from the given object
            traverse(obj)

            # Calculate combined dimensions
            combined_dimensions = bbox_max - bbox_min

            return combined_dimensions

        # Calculate combined dimensions of all mesh objects in the hierarchy
        mesh_dimensions = calculate_combined_dimensions(gltf_obj)
        print(f"Combined dimensions of  mesh: {mesh_dimensions}")
        return list(mesh_dimensions)
    else:
        print("Object not found in the scene.")


def calculate_scale_to_match_dimensions(source_obj, target_obj):
    # Get dimensions of the source object
    # source_dimensions = calculate_dimensions(source_obj)
    bpy.context.view_layer.update()
    source_dimensions = get_dimensions_for_hierarchical_mesh(source_obj)
    # source_dimensions = source_obj.dimensions
    print(source_dimensions)

    # Get dimensions of the target object
    target_dimensions = target_obj.dimensions
    print(target_dimensions)
    # Calculate scale factors for each axis
    scale_factors = (
        target_dimensions.x / source_dimensions[0],
        target_dimensions.y / source_dimensions[1],
        target_dimensions.z / source_dimensions[2],
    )

    min_scale = min(scale_factors[0], scale_factors[1])
    min_scale = min(min_scale, scale_factors[2])
    min_scale = abs(min_scale)
    scale_factors = (min_scale, min_scale, min_scale)

    return scale_factors


def make_object_invisible(obj):
    # Hide the object in the viewport (for visibility in the 3D Viewport)
    obj.hide_viewport = True

    # Hide the object during rendering (for visibility in final renders)
    obj.hide_render = True


def find_ground_plane():
    # Get all mesh objects in the scene
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    # Initialize variables for tracking the ground plane
    ground_object = None
    min_z = float("inf")

    # Iterate through mesh objects to find the lowest object (potential ground plane)
    for obj in mesh_objects:
        bbox = obj.bound_box
        z_coords = [obj.matrix_world @ mathutils.Vector(v) for v in bbox]
        min_obj_z = min(v.z for v in z_coords)

        if min_obj_z < min_z:
            min_z = min_obj_z
            ground_object = obj

    # Print information about the identified ground plane
    if ground_object:
        print(f"Ground Plane Object: {ground_object.name}")
        print(f"Minimum Z-coordinate: {min_z}")
        return min_z
    else:
        print("No ground plane found")


def get_bbox_hierarchical_mesh(gltf_obj):
    if gltf_obj:
        # Update the object's bounding box to ensure dimensions are calculated
        bpy.context.view_layer.update()

        # Function to recursively calculate combined dimensions of all mesh objects in the hierarchy
        def calculate_combined_dimensions(obj):
            # Initialize variables to track combined dimensions
            bbox_min = mathutils.Vector((float("inf"), float("inf"), float("inf")))
            bbox_max = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))

            # Traverse through all children recursively
            def traverse(obj):
                nonlocal bbox_min, bbox_max

                # Check if the object is a mesh with valid mesh data
                if obj.type == "MESH" and obj.data:
                    # Get local bounding box dimensions
                    mesh_bbox_min, mesh_bbox_max = obj.bound_box[0], obj.bound_box[6]
                    # Apply object's local transformation
                    # import pdb; pdb.set_trace()
                    mesh_bbox_min = obj.matrix_world @ mathutils.Vector(mesh_bbox_min)
                    mesh_bbox_max = obj.matrix_world @ mathutils.Vector(mesh_bbox_max)

                    # Update combined dimensions
                    bbox_min = mathutils.Vector(
                        (
                            min(bbox_min.x, mesh_bbox_min.x),
                            min(bbox_min.y, mesh_bbox_min.y),
                            min(bbox_min.z, mesh_bbox_min.z),
                        )
                    )
                    bbox_max = mathutils.Vector(
                        (
                            max(bbox_max.x, mesh_bbox_max.x),
                            max(bbox_max.y, mesh_bbox_max.y),
                            max(bbox_max.z, mesh_bbox_max.z),
                        )
                    )

                # Recursively traverse children
                for child in obj.children:
                    traverse(child)

            # Start traversal from the given object
            traverse(obj)

            return bbox_min, bbox_max

        # Calculate combined dimensions of all mesh objects in the hierarchy
        min_, max_ = calculate_combined_dimensions(gltf_obj)
        return list(min_), list(max_)
    else:
        print("Object not found in the scene.")


print("output-dir", sys.argv[-1])
print("glb path", sys.argv[-2])


# Path of the glb mesh from objaverse
path_to_object_model = str(sys.argv[-2])
output_directory = str(sys.argv[-1])
os.makedirs(output_directory, exist_ok=True)

# Get mirror location
mirror_obj = bpy.data.objects["Cube.001"]
mirror_location = mirror_obj.location

# Above mirror object for this scene is not making sense hard-cpde
# Print the coordinates

mirror_location = (-5.6997, 1.3057, 0.75271)
print(
    f"Mirror Location '{mirror_obj.name}' is located at (x={mirror_location[0]}, y={mirror_location[1]}, z={mirror_location[2]})"
)

# Location in front of the mirror
cube_location = bpy.data.objects["Cylinder"].location

# Now set Camera location
camera_obj = bpy.data.objects["Camera"]
# Set the location of the camera object
# camera_obj.location = (-3.4267, -1.4192, 1.9165)

# new_camera_location = shift_location( camera_obj.location , -1.5 )
# camera_obj.location = new_camera_location
print("Camera location", camera_obj.location)

# Sample new camera locations on a spherical manifold
cam_locations = sample_camera_locations(mirror_location, camera_obj.location)

scene_min_z = find_ground_plane()

# Check if the file exists
if os.path.exists(path_to_object_model):
    # Import the OBJ file
    bpy.ops.import_scene.gltf(filepath=path_to_object_model)

    # Optionally, adjust the imported object's location, rotation, scale, etc.
    imported_object = bpy.context.active_object
    # imported_object.location = cube_location #(cube_location[0]+0.25, cube_location[1], cube_location[2])  # Set the location of the imported object
    imported_object.location = (-4.5864, 0.77726, 0.010368)

    # Create a Quaternion object from specified values
    rotation_quaternion = Quaternion((0.303141, -0.303141, 0.638832, -0.638832))

    # Set the object's rotation using quaternion
    imported_object.rotation_quaternion = rotation_quaternion

    # imported_object.rotation_euler = (0, 90, 0)  # Set the rotation of the imported object

    scale_val = calculate_scale_to_match_dimensions(
        imported_object, bpy.data.objects["Cylinder"]
    )
    print("Estimated scale is ", scale_val)
    imported_object.scale = scale_val  # Set the scale of the imported object
    obj_min, obj_max = get_bbox_hierarchical_mesh(imported_object)
    print("Object min bound ", obj_min)
    print("Object max bound ", obj_max)
    if obj_min[2] < 0:
        print("Object beneath the ground")
        diff = 0.01 - obj_min[2]
        imported_object.location = (-4.5864, 0.77726, 0.010368 + diff)
        obj_min, obj_max = get_bbox_hierarchical_mesh(imported_object)
        print("After correction Object min bound ", obj_min)
        print("After correction Object max bound ", obj_max)


make_object_invisible(bpy.data.objects["Cylinder"])
# make_object_invisible(bpy.data.objects['Sketchfab_model.001'])
# Set rendering resolution and output settings
bpy.context.scene.render.resolution_x = 640
bpy.context.scene.render.resolution_y = 480
bpy.context.scene.render.image_settings.file_format = "PNG"

# Enable ray tracing and adjust reflection settings
bpy.context.scene.render.engine = (
    "CYCLES"  # Use Cycles rendering engine for realistic reflections
)

bpy.context.scene.render.resolution_percentage = 100
bpy.context.scene.cycles.samples = 2000  # Increase samples for better quality
bpy.context.scene.cycles.max_bounces = 96  # Increase max bounces for reflections
bpy.context.scene.view_layers[0].cycles.use_denoising = True


# Enable ambient occlusion
# bpy.context.scene.render.layers[0].cycles.use_ambient_occlusion = True

# Iterate over each camera location and render
for i, location in enumerate(cam_locations, start=1):
    print(location)
    # Set camera location
    camera_obj.location = location

    # Set output file path for the rendered image
    filepath = os.path.join(output_directory, f"render_{i:03d}.png")

    # Set output file path
    bpy.context.scene.render.filepath = filepath

    # Render the scene
    start_time = time.time()
    bpy.ops.render.render(write_still=True)
    elapsed_time = time.time() - start_time

    # Print the elapsed time
    print(f"Elapsed time: {elapsed_time:.6f} seconds")