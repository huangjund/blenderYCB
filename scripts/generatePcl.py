import argparse
import json
import os
import trimesh
import numpy as np

def sample_point_cloud(mesh_path, num_points):
    mesh = trimesh.load(mesh_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Failed to load a mesh from {mesh_path}")
    points = mesh.sample(num_points)
    return points

def save_xyz(points, save_path):
    np.savetxt(save_path, points, fmt="%.6f")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/config.json", help="Path to the configuration file")
    args = parser.parse_args()

    # Load the configuration JSON
    with open(args.config, 'r') as f:
        config = json.load(f)

    models_path = os.path.join("..",config['models_path'])
    num_points = config['num_pts']

    for filename in os.listdir(models_path):
        if filename.endswith(".obj"):
            obj_path = os.path.join(models_path, filename)
            point_cloud = sample_point_cloud(obj_path, num_points)

            xyz_filename = os.path.splitext(filename)[0] + ".xyz"
            xyz_path = os.path.join(models_path, xyz_filename)
            save_xyz(point_cloud, xyz_path)
            print(f"Saved point cloud to {xyz_path}")
