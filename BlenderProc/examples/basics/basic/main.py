import blenderproc as bproc
from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects
import argparse

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
debugpy.breakpoint()

parser = argparse.ArgumentParser()
parser.add_argument('camera', help="Path to the camera file, should be examples/resources/camera_positions")
parser.add_argument('scene', help="Path to the scene.obj file, should be examples/resources/scene.obj")
parser.add_argument('output_dir', help="Path to where the final files, will be saved, could be examples/basics/basic/output")
args = parser.parse_args()

bproc.init()

# load the objects into the scene
objs = bproc.loader.load_obj(args.scene)
for j, obj in enumerate(objs):
    print("Setting category id", j + 1)
    obj.set_cp("category_id", j + 1)

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([5, -5, 5])
light.set_energy(1000)

# define the camera resolution
bproc.camera.set_resolution(512, 512)

# read the camera positions file and convert into homogeneous camera-world transformation
with open(args.camera, "r") as f:
    for line in f.readlines():
        line = [float(x) for x in line.split()]
        position, euler_rotation = line[:3], line[3:6]
        matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)
        bproc.camera.add_camera_pose(matrix_world)

# activate normal and depth rendering
# bproc.renderer.enable_normals_output()
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(default_values={"category_id": 0})

# render the whole pipeline
data = bproc.renderer.render()

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
