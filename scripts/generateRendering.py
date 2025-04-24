import blenderproc as bproc
import argparse
import os
import math
import numpy as np
import h5py
import json
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import sys
import bpy
import h5py
import scipy.io as sio  # Import SciPy for saving .mat files
from blenderproc.python.utility.CollisionUtility import CollisionUtility

# sys.path.append("/home/jd/miniconda3/envs/render/lib/python3.10/site-packages/")
# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

rot_x = lambda deg: R.from_euler('x', np.radians(deg)).as_matrix()
rot_y = lambda deg: R.from_euler('y', np.radians(deg)).as_matrix()
rot_z = lambda deg: R.from_euler('z', np.radians(deg)).as_matrix()


class BlenderProcRenderer:
    def __init__(self, config_path):
        self.load_config(config_path)
        self.kinematic_chain = self.read_kinematics_from_urdf(self.config["urdf_path"])
        np.random.seed(self.config["numpy_seed"])
        bproc.init()
        self.objs = {}
        self.scene_files= []

    def load_config(self, config_path):
        with open(config_path, "r") as f:
            self.config = json.load(f)

    def create_transformation_matrix(self, xyz, rpy=np.array([1.57, 0, 0])):
        xyz = np.array(xyz).reshape(3, 1)
        R_matrix = R.from_euler("xyz", rpy, degrees=False).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R_matrix
        T[:3, 3] = xyz.flatten()
        return T

    def read_kinematics_from_urdf(self, urdf_path):
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
        except (ET.ParseError, FileNotFoundError) as e:
            print(f"Error reading URDF: {e}")
            return {}

        kinematic_data = {}
        for joint in root.findall("joint"):
            name = joint.get("name", "Unknown")
            joint_type = joint.get("type", "Unknown")
            parent = joint.find("parent").get("link", "Unknown") if joint.find("parent") is not None else "Unknown"
            child = joint.find("child").get("link", "Unknown") if joint.find("child") is not None else "Unknown"
            origin_element = joint.find("origin")
            xyz = np.array(list(map(float, (origin_element.get("xyz", "0 0 0") if origin_element is not None else "0 0 0").split())))
            rpy = np.array(list(map(float, (origin_element.get("rpy", "0 0 0") if origin_element is not None else "0 0 0").split())))
            axis = np.array(list(map(float, (joint.find("axis").get("xyz", "0 0 0") if joint.find("axis") is not None else "0 0 0").split())))

            transformation_matrix = self.create_transformation_matrix(xyz, rpy)
            kinematic_data[name] = {
                "type": joint_type,
                "parent": parent,
                "child": child,
                "origin": {"xyz": xyz, "rpy": rpy},
                "axis": axis,
                "transformation_matrix": transformation_matrix,
            }
        return kinematic_data

    def load_scenes(self):
        scene_dir = os.path.join(self.config["scene"]["path"], "JPEGImages")
        if os.path.exists(scene_dir) and os.path.isdir(scene_dir):
            self.scene_files = [os.path.join(scene_dir, f) for f in os.listdir(scene_dir) if
                           os.path.isfile(os.path.join(scene_dir, f))]
            print("Scene directory files:", len(self.scene_files))

    def load_models(self):
        obj_dir = self.config["models_path"]
        if os.path.exists(obj_dir) and os.path.isdir(obj_dir):
            obj_files = [os.path.join(obj_dir, f) for f in os.listdir(obj_dir) if f.endswith(".obj")]
            self.objs = {os.path.basename(obj_file): bproc.loader.load_obj(obj_file) for obj_file in obj_files}
        for i, obj in enumerate(self.objs):
            if obj == 'base_link.obj':
                self.objs[obj][0].set_cp("category_id",0)
            else:
                self.objs[obj][0].set_cp("category_id",i)
    def setup_camera(self):
        cam_config = self.config["camera"]
        self.intrinsic_matrix = np.array([
            [cam_config["px"], 0, cam_config["u0"]],
            [0, cam_config["py"], cam_config["v0"]],
            [0, 0, 1]
        ])
        bproc.camera.set_intrinsics_from_K_matrix(self.intrinsic_matrix, cam_config["w"], cam_config["h"])

    def setup_light(self):
        # define a light and set its location and energy level
        light = bproc.types.Light()
        light.set_type(self.config["light"]["type"])
        light.set_location(self.config["light"]["location"])
        light.set_energy(self.config["light"]["energy"])

    def create_room(self):
        size = self.config["scene"]["room_size"] * np.random.uniform(self.config["scene"]["room_size_multiplier_min"], self.config["scene"]["room_size_multiplier_max"])
        ground = bproc.object.create_primitive("PLANE")
        ground.set_location([0, 0.3, 0.5])
        ground.set_rotation_euler([np.radians(-200), 0, 0])
        ground.set_scale([size, size, 1])
        ground.set_cp("category_id",63)
        return [ground]

    def get_object_image_coordinates(self):
        """
        Computes the 3D bounding box center of each object and projects it onto the image plane.
        """
        centers_2d = {}  # Dictionary to store object names and their projected coordinates

        for i, obj in enumerate(self.objs):
            if obj != "plane":
                # Compute 3D bounding box center
                bbox = self.objs[obj][0].get_bound_box()
                bbox_center_3d = np.mean(bbox, axis=0)  # Compute center of bounding box

                # Project 3D point to image plane and replace the value < 0 to 1
                image_coords = np.maximum(bproc.camera.project_points(bbox_center_3d.reshape(1,3)), 1)

                if image_coords is not None:
                    centers_2d[self.objs[obj][0].get_cp("category_id")] = image_coords  # Store projected coordinates

        return centers_2d

    def get_object_poses(self):
        """
        Retrieves the 4x4 homogeneous transformation matrix of each object in the scene.
        """
        object_poses = {}

        for i, obj in enumerate(self.objs):
            # Get 4x4 homogeneous transformation matrix
            transformation_matrix = self.objs[obj][0].get_local2world_mat()

            # Store the transformation matrix
            object_poses[self.objs[obj][0].get_cp("category_id")] = transformation_matrix[:3,:4]

        return object_poses

    def check_collisions(self):
        """
        Checks for collisions between a main object and a list of scene objects.

        :return: A boolean to indicate wether collision happens where keys are object names and values are boolean indicating collision.
        """
        # Initialize BVH cache
        bvh_L3R3_cache = {}
        bvh_L3R2_cache = {}
        bvh_L2R3_cache = {}
        bvh_L2R2_cache = {}

        L2 = self.objs['L2.obj'][0]
        L3 = self.objs['L3.obj'][0]
        R2 = self.objs['R2.obj'][0]
        R3 = self.objs['R3.obj'][0]

        collisions = False

        # Check mesh intersection
        collisions, updated_bvh_L3R3_cache = CollisionUtility.check_mesh_intersection(
            L3,
            R3,
            skip_inside_check=True,
            bvh_cache=bvh_L3R3_cache
        )
        if collisions:
            return collisions
        collisions, updated_bvh_L3R2_cache = CollisionUtility.check_mesh_intersection(
            L3,
            R2,
            skip_inside_check=True,
            bvh_cache=bvh_L3R2_cache
        )
        if collisions:
            return collisions
        collisions, updated_bvh_L2R3_cache = CollisionUtility.check_mesh_intersection(
            L2,
            R3,
            skip_inside_check=True,
            bvh_cache=bvh_L2R3_cache
        )
        if collisions:
            return collisions
        collisions, updated_bvh_L2R2_cache = CollisionUtility.check_mesh_intersection(
            L2,
            R2,
            skip_inside_check=True,
            bvh_cache=bvh_L2R2_cache
        )

        return collisions

    def render_scene(self):
        self.setup_camera()
        self.load_scenes()
        self.load_models()
        self.setup_light()
        # room_obj = self.create_room()
        scene_output = self.config["dataset"]["save_path"]

        motion_range_bound_low = self.config["object"]["motion_range_bound_low"]
        motion_range_bound_high = self.config["object"]["motion_range_bound_high"]

        scene_nums = np.array(self.config["dataset"]["scenes_per_run"])
        num_frames = np.array(self.config["dataset"]["images_per_scene"])

        np.random.seed(self.config["numpy_seed"])
        selected_scenes = list(np.random.choice(self.scene_files, scene_nums, replace=False))  # No duplicates
        for i, bg_image in enumerate(selected_scenes):
            print(f"Rendering scene {i + 1}/{scene_nums}")
            # Clear keyframes
            bproc.utility.reset_keyframes()

            # create room with different size
            # room_obj[0].delete()
            # room_obj = self.create_room()
            # bg_img_obj = bpy.data.images.load(filepath=bg_image)
            # material = bproc.material.create(name=bg_image)
            # material.set_principled_shader_value("Base Color", bg_img_obj)
            # # room_obj[0].set_shading_mode("SMOOTH")
            # room_obj[0].replace_materials(material)
            # bg_image

            # creat background
            rnd = np.random.uniform(0,1)
            if rnd < 0:
                bproc.world.set_world_background_hdr_img(bg_image)
            else:
                bproc.world.set_world_background_hdr_img(os.path.join(self.config["scene"]["path"], "Test2.jpg"),
                                                         rotation_euler=np.random.uniform([np.pi/2, 0, 0],[np.pi*3/2, 2*np.pi, 2*np.pi]))

            for frame in range(num_frames):
                bproc.camera.add_camera_pose(bproc.math.build_transformation_mat(
                    self.config["camera"]["location"]["xyz"],np.radians(self.config["camera"]["location"]["rpy"]),
                ))
            if i+1 == 1:
                bproc.renderer.enable_normals_output()
                bproc.renderer.enable_depth_output(activate_antialiasing=False)
                bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

            for frame in range(num_frames):
                while 1:                # Set keyframe for object position at this frame
                    w1 = self.config["object"]["collision_w"][0]
                    w2 = self.config["object"]["collision_w"][1]
                    while True:
                        joint_angles = np.random.uniform(motion_range_bound_low, motion_range_bound_high)
                        # Check collision
                        if (w1 * (joint_angles[1] + joint_angles[4]) + w2 * (joint_angles[2] + joint_angles[5]) <= 60 and
                                (joint_angles[1] + joint_angles[4] <= 30)):
                            joint_angles = self.config["object"]["direction"] * joint_angles
                            break  # Exit loop when both constraints are met
                    # joint_angles = np.random.uniform(motion_range_bound_low, motion_range_bound_high)
                    # joint_angles = self.config["object"]["direction"] * joint_angles
                    # set objs poses
                    T01_L = self.kinematic_chain['L1_joint']['transformation_matrix'] @ self.create_transformation_matrix(
                        np.array([0, 0, 0]), np.radians(np.array([0, 0, -1 * joint_angles[0]])))
                    T12_L = self.kinematic_chain['L2_joint']['transformation_matrix'] @ self.create_transformation_matrix(
                        np.array([0, 0, 0]), np.radians(np.array([0, 0, -1 * joint_angles[1]])))
                    T23_L = self.kinematic_chain['L3_joint']['transformation_matrix'] @ self.create_transformation_matrix(
                        np.array([0, 0, 0]), np.radians(np.array([0, 0, -1 * joint_angles[2]])))
                    L2_offset = np.eye(4)  # to adjust the intrinsic offset
                    L2_offset[:3, :3] = rot_x(90)
                    L3_offset = np.eye(4)
                    L3_offset[:3, :3] = rot_x(90)
                    T02_L = T01_L @ T12_L @ L2_offset
                    T03_L = T01_L @ T12_L @ T23_L @ L3_offset
                    self.objs['L2.obj'][0].set_location(T02_L[:3, 3], frame=frame)
                    self.objs['L2.obj'][0].set_rotation_mat(T02_L[:3, :3], frame=frame)
                    self.objs['L3.obj'][0].set_location(T03_L[:3, 3], frame=frame)
                    self.objs['L3.obj'][0].set_rotation_mat(T03_L[:3, :3], frame=frame)

                    T01_R = self.kinematic_chain['R1_joint']['transformation_matrix'] @ self.create_transformation_matrix(
                        np.array([0, 0, 0]), np.radians(np.array([0, 0, -1 * joint_angles[3]])))
                    T12_R = self.kinematic_chain['R2_joint']['transformation_matrix'] @ self.create_transformation_matrix(
                        np.array([0, 0, 0]), np.radians(np.array([0, 0, -1 * joint_angles[4]])))
                    T23_R = self.kinematic_chain['R3_joint']['transformation_matrix'] @ self.create_transformation_matrix(
                        np.array([0, 0, 0]), np.radians(np.array([0, 0, -1 * joint_angles[5]])))
                    R2_offset = np.eye(4)
                    R2_offset[:3, :3] = rot_x(90)
                    R3_offset = np.eye(4)
                    R3_offset[:3, :3] = rot_x(90)
                    T02_R = T01_R @ T12_R @ R2_offset
                    T03_R = T01_R @ T12_R @ T23_R @ R3_offset
                    self.objs['R2.obj'][0].set_location(T02_R[:3, 3], frame=frame)
                    self.objs['R2.obj'][0].set_rotation_mat(T02_R[:3, :3], frame=frame)
                    self.objs['R3.obj'][0].set_location(T03_R[:3, 3], frame=frame)
                    self.objs['R3.obj'][0].set_rotation_mat(T03_R[:3, :3], frame=frame)

                    # Collision checking, if no collision, continue
                    if self.check_collisions() == False:
                        break

                centers = self.get_object_image_coordinates()
                cls_indexes = np.array(list(centers.keys()), dtype=np.int32)  # Extract category IDs
                poses = self.get_object_poses()

                # remove base_link
                valid_mask = cls_indexes != self.objs['base_link.obj'][0].get_cp("category_id")
                valid_indices = np.where(valid_mask)[0]

                meta_data = {
                    "center": np.array(list(centers.values()), dtype=np.float32).squeeze(axis=1)[valid_mask],  # (N, 2)
                    "cls_indexes": cls_indexes[valid_mask],  # (N,)
                    "intrinsic_matrix": np.array(self.intrinsic_matrix, dtype=np.float64),  # (3,3)
                    "poses": np.transpose(np.array([poses[cls] for cls in cls_indexes[valid_mask]], dtype=np.float32), (1, 2, 0)),  # (3, 4, N)
                    "factor_depth": 10000
                }

                os.makedirs(os.path.join(scene_output, f"rendered_scene_{i + 1}"), exist_ok=True)
                meta_path = os.path.join(scene_output, f"rendered_scene_{i + 1}",f"{frame}_meta.mat")
                sio.savemat(meta_path, meta_data)

            data = bproc.renderer.render()
            output_path = os.path.join(scene_output, f"rendered_scene_{i + 1}")
            data["depth"] = [element*10000 for element in data["depth"]]
            bproc.writer.write_hdf5(output_path, data)
            print(f"Saved rendered images to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.json", help="Path to the configuration file")
    args = parser.parse_args()

    renderer = BlenderProcRenderer(args.config)
    renderer.render_scene()




# def randomize_pbr(objects: List[bproc.types.MeshObject], noise):
#   '''
#   Randomize the physical properties of objects.
#   Objects should have the Principled BSDF for changes to be applied.
#
#   '''
#   # set shading and physics properties and randomize PBR materials
#   for j, obj in enumerate(objects):
#     # rand_angle = np.random.uniform(30, 90)
#     # obj.set_shading_mode('auto', rand_angle)
#     for mat in obj.get_materials():
#       keys = ["Roughness", "Specular", "Metallic"]
#       for k in keys:
#         base_value = mat.get_principled_shader_value(k)
#         new_value = None
#         if isinstance(base_value, bpy.types.NodeSocketColor):
#           new_value = base_value.default_value + np.concatenate((np.random.uniform(-noise, noise, size=3), [1.0]))
#           #base_value.default_value = new_value
#           new_value = base_value
#         elif isinstance(base_value, float):
#           new_value = max(0.0, min(1.0, base_value + np.random.uniform(-noise, noise)))
#         if new_value is not None:
#           mat.set_principled_shader_value(k, new_value)
