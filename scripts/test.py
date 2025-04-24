import blenderproc as bproc
import argparse
import os
import math
import numpy as np
import h5py
import json
import xml.etree.ElementTree as ET # read urdf
from scipy.spatial.transform import Rotation as R
import sys
import json
import cv2
from typing import List
import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--camera', help="Path to the camera file, should be examples/resources/camera_positions")
parser.add_argument('--obj', default="data/objs_custom_data", help="Path to the scene.obj folder, should be examples/resources/scene.obj")
parser.add_argument('--scene', default="data/sun2012pascalformat", help="Path to the scene folder, should be examples/resources/scene")
parser.add_argument('--output_dir', default="output", help="Path to where the final files will be saved, could be examples/basics/basic/output")
args = parser.parse_args()

rot_x = lambda deg: R.from_euler('x', np.radians(deg)).as_matrix()
rot_y = lambda deg: R.from_euler('y', np.radians(deg)).as_matrix()
rot_z = lambda deg: R.from_euler('z', np.radians(deg)).as_matrix()

scene_dir = os.path.join(args.scene, "JPEGImages")
if os.path.exists(scene_dir) and os.path.isdir(scene_dir):
    scene_files = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if os.path.isfile(os.path.join(scene_dir, f))]
    print("Scene directory files:", len(scene_files))
else:
    print(f"Scene directory '{scene_dir}' not found.")

obj_dir = args.obj
if os.path.exists(obj_dir) and os.path.isdir(obj_dir):
    obj_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith(".obj") and os.path.isfile(os.path.join(obj_dir, f))]
    print("Number of .obj files:", len(obj_files))
else:
    print(f"OBJ directory '{obj_dir}' not found.")

def create_transformation_matrix(xyz, rpy=np.array([1.57, 0, 0])):
    """Create a 4x4 transformation matrix from position (xyz) and rotation (rpy)"""
    xyz = np.array(xyz).reshape(3, 1)  # Convert to column vector
    R_matrix = R.from_euler('xyz', rpy, degrees=False).as_matrix()  # Convert RPY to rotation matrix

    T = np.eye(4)  # Identity matrix
    T[:3, :3] = R_matrix  # Set rotation
    T[:3, 3] = xyz.flatten()  # Set translation
    return T

def read_kinematics_from_urdf(urdf_path):
    """Reads a URDF file and extracts kinematic transformations."""
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing URDF file: {e}")
        return
    except FileNotFoundError:
        print(f"URDF file not found: {urdf_path}")
        return

    print(f"Parsing URDF: {urdf_path}\n")

    # Iterate over all joint elements
    kinematic_data = {}
    for joint in root.findall('joint'):
        name = joint.get('name', 'Unknown')
        joint_type = joint.get('type', 'Unknown')
        parent = joint.find('parent').get('link', 'Unknown') if joint.find('parent') is not None else 'Unknown'
        child = joint.find('child').get('link', 'Unknown') if joint.find('child') is not None else 'Unknown'

        # Extract transformation (origin)
        origin_element = joint.find('origin')
        xyz = np.array(list(map(float, (origin_element.get('xyz', '0 0 0') if origin_element is not None else '0 0 0').split())))
        rpy = np.array(list(map(float, (origin_element.get('rpy', '0 0 0') if origin_element is not None else '0 0 0').split())))

        # Extract rotation axis
        axis = np.array(list(map(float, (joint.find('axis').get('xyz', '0 0 0') if joint.find('axis') is not None else '0 0 0').split())))

        transformation_matrix = create_transformation_matrix(xyz, rpy)

        # Store data in dictionary
        kinematic_data[name] = {
            "type": joint_type,
            "parent": parent,
            "child": child,
            "origin": {
                "xyz": xyz,
                "rpy": rpy
            },
            "axis": axis,
            "transformation_matrix": transformation_matrix  # Convert NumPy array to list for JSON compatibility
        }
    return kinematic_data

def readh5(filename):
    with h5py.File(filename) as f:
        colors = np.array(f["colors"])
        text = np.array(f["object_states"]).tostring()
        obj_states = json.loads(text)
    return colors

def getIntrinsic(json_path, resolution=(640, 480)):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Determine the correct intrinsic parameters based on resolution
    for i in range(16):  # There are multiple resolutions in the data
        key_prefix = f"rectified.{i}"
        if key_prefix + ".width" in data and key_prefix + ".height" in data:
            width = int(data[key_prefix + ".width"])
            height = int(data[key_prefix + ".height"])
            if (width, height) == resolution:
                fx = float(data[key_prefix + ".fx"])
                fy = float(data[key_prefix + ".fy"])
                ppx = float(data[key_prefix + ".ppx"])
                ppy = float(data[key_prefix + ".ppy"])

                # Construct the intrinsic matrix
                intrinsic_matrix = np.array([
                    [fx, 0, ppx],
                    [0, fy, ppy],
                    [0, 0, 1]
                ])

                return intrinsic_matrix

    # If no matching resolution is found, return None
    print("Resolution not found in JSON data.")
    return None

def randomize_pbr(objects: List[bproc.types.MeshObject], noise):
  '''
  Randomize the physical properties of objects.
  Objects should have the Principled BSDF for changes to be applied.
  '''
  # set shading and physics properties and randomize PBR materials
  for j, obj in enumerate(objects):
    # rand_angle = np.random.uniform(30, 90)
    # obj.set_shading_mode('auto', rand_angle)
    for mat in obj.get_materials():
      keys = ["Roughness", "Specular", "Metallic"]
      for k in keys:
        base_value = mat.get_principled_shader_value(k)
        new_value = None
        if isinstance(base_value, bpy.types.NodeSocketColor):
          new_value = base_value.default_value + np.concatenate((np.random.uniform(-noise, noise, size=3), [1.0]))
          #base_value.default_value = new_value
          new_value = base_value
        elif isinstance(base_value, float):
          new_value = max(0.0, min(1.0, base_value + np.random.uniform(-noise, noise)))
        if new_value is not None:
          mat.set_principled_shader_value(k, new_value)

def create_room():
    '''
    Create a basic scene, a square room.
    The size of the room is dependent on the size of the biggest object multiplied by a user supplied param.
    Each wall has a random texture, sampled from cc0 materials.
    Distractors are added randomly in the room.
    Random lights are also placed.
    If simulate physics is true, then objects are dropped on the ground.
    When using simulate physics, some objects may leave the room (through weird collisions): they are deleted.

    '''
    room_size = 0.4
    room_size_multiplier_min, room_size_multiplier_max = 1,2

    size = room_size * np.random.uniform(room_size_multiplier_min, room_size_multiplier_max)

    ground = bproc.object.create_primitive('PLANE')
    ground.set_location([0, size/3, size/2])
    ground.set_rotation_euler([np.radians(-200), 0, 0])
    ground.set_scale([size/2, size/2, 1])
    room_objects = [ground]

    # for obj in room_objects:
    #   random_texture = np.random.choice(self.cc_textures)
    #   obj.replace_materials(random_texture)
    # randomize_pbr(room_objects, 0.2)

    return room_objects


# read in kinematics
urdf_file = "data/AssemV_camera/urdf/AssemV_camera.urdf"
kin_chain = read_kinematics_from_urdf(urdf_file)

bproc.init()

# Store loaded objects in a dictionary (key: file name, value: loaded object)
objs = {os.path.basename(obj_file): bproc.loader.load_obj(obj_file) for obj_file in obj_files}

# define a light and set its location and energy level
light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0, 2, -2])
light.set_energy(1000)

# define the camera resolution and configure
ycb_K = getIntrinsic(os.path.join("data/AssemV_camera/","d405.json"))
print(ycb_K)
bproc.camera.set_intrinsics_from_K_matrix(ycb_K,640, 480)

# read the camera positions file and convert into homogeneous camera-world transformation
line = [0, 0.05, -0.16, math.radians(-200), 0, 0]
position, euler_rotation = line[:3], line[3:6]
matrix_world = bproc.math.build_transformation_mat(position, euler_rotation)

# start rendering
scene_nums = 1
scene_seed = 42
motion_range_bound_low = np.array([-20, -20, -20, -20, -20, -20])
motion_range_bound_up = np.array([20, 20, 20, 20, 20, 20])
num_frames = 2
np.random.seed(scene_seed)
selected_scenes = list(np.random.choice(scene_files, scene_nums, replace=False))  # No duplicates

# Render for each background image
for i, bg_image in enumerate(selected_scenes):
    print(f"Rendering scene {i + 1}/{len(scene_files)} with background: {bg_image}")
    # Clear keyframes
    bproc.utility.reset_keyframes()

    # # Set environment texture as background
    # bproc.world.set_world_background_hdr_img(bg_image)
    room_obj = create_room()
    bg_img_obj = bpy.data.images.load(filepath=bg_image)
    material = bproc.material.create(name=bg_image)
    material.set_principled_shader_value("Base Color", bg_img_obj)
    # room_obj[0].set_shading_mode("SMOOTH")
    room_obj[0].replace_materials(material)

    # Set keyframes for animation
    for frame in range(num_frames):
        # Set camera
        bproc.camera.add_camera_pose(matrix_world)
    # activate normal and depth rendering
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    for frame in range(num_frames):
        # Set keyframe for object position at this frame
        joint_angles = np.random.uniform(motion_range_bound_low, motion_range_bound_up)
        # set objs poses
        T01_L = kin_chain['L1_joint']['transformation_matrix'] @ create_transformation_matrix(np.array([0,0,0]), np.radians(np.array([0,0,-1*joint_angles[0]])))
        T12_L = kin_chain['L2_joint']['transformation_matrix'] @ create_transformation_matrix(np.array([0,0,0]), np.radians(np.array([0,0,-1*joint_angles[1]])))
        T23_L = kin_chain['L3_joint']['transformation_matrix'] @ create_transformation_matrix(np.array([0,0,0]), np.radians(np.array([0,0,-1*joint_angles[2]])))
        L2_offset = np.eye(4) # to adjust the intrinsic offset
        L2_offset[:3, :3] = rot_x(90)
        L3_offset = np.eye(4)
        L3_offset[:3, :3] = rot_x(90)
        T02_L = T01_L @ T12_L @ L2_offset
        T03_L = T01_L @ T12_L @ T23_L @ L3_offset
        objs['L2.obj'][0].set_location(T02_L[:3,3],frame=frame)
        objs['L2.obj'][0].set_rotation_mat(T02_L[:3, :3], frame=frame)
        objs['L3.obj'][0].set_location(T03_L[:3,3],frame=frame)
        objs['L3.obj'][0].set_rotation_mat(T03_L[:3, :3], frame=frame)

        T01_R = kin_chain['R1_joint']['transformation_matrix'] @ create_transformation_matrix(np.array([0,0,0]), np.radians(np.array([0,0,-1*joint_angles[3]])))
        T12_R = kin_chain['R2_joint']['transformation_matrix'] @ create_transformation_matrix(np.array([0,0,0]), np.radians(np.array([0,0,-1*joint_angles[4]])))
        T23_R = kin_chain['R3_joint']['transformation_matrix'] @ create_transformation_matrix(np.array([0,0,0]), np.radians(np.array([0,0,-1*joint_angles[5]])))
        R2_offset = np.eye(4)
        R2_offset[:3, :3] = rot_x(90)
        R3_offset = np.eye(4)
        R3_offset[:3, :3] = rot_x(90)
        T02_R = T01_R @ T12_R @ R2_offset
        T03_R = T01_R @ T12_R @ T23_R @ R3_offset
        objs['R2.obj'][0].set_location(T02_R[:3,3],frame=frame)
        objs['R2.obj'][0].set_rotation_mat(T02_R[:3, :3], frame=frame)
        objs['R3.obj'][0].set_location(T03_R[:3,3],frame=frame)
        objs['R3.obj'][0].set_rotation_mat(T03_R[:3, :3], frame=frame)

    # Render each frame
    data = bproc.renderer.render()
    output_path = os.path.join(args.output_dir, f"rendered_scene_{i + 1}")
    bproc.writer.write_hdf5(output_path, data)
    print(f"Saved rendered image to {output_path}")
