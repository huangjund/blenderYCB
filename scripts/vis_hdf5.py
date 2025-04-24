import os
import h5py
import numpy as np
import cv2

def visualize_hdf5_files(base_folder):
    """
    Iterates through all 'rendered_scene_*' folders and visualizes HDF5 files using OpenCV.

    Parameters:
        base_folder (str): Path to the folder containing rendered scene folders.
    """
    # Get all rendered scene folders
    scene_folders = sorted([f for f in os.listdir(base_folder) if f.startswith("rendered_scene_")])

    if not scene_folders:
        print("No 'rendered_scene_*' folders found.")
        return

    for scene_folder in scene_folders:
        scene_path = os.path.join(base_folder, scene_folder)
        print(f"\n🟢 Processing Scene: {scene_folder}")

        # Get all HDF5 files in the current scene folder
        hdf5_files = sorted([f for f in os.listdir(scene_path) if f.endswith(".hdf5")])

        if not hdf5_files:
            print(f"⚠ No HDF5 files found in {scene_folder}. Skipping...")
            continue

        for hdf5_file in hdf5_files:
            file_path = os.path.join(scene_path, hdf5_file)
            print(f"\n🔹 Processing File: {hdf5_file}")

            with h5py.File(file_path, "r") as f:
                # 🔹 Display RGB Images  
                if "colors" in f:
                    img = f["colors"]
                    rgb_img = np.array(img)
                    cv2.imshow(f"RGB Image - {hdf5_file} ", cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(0)

                # 🔹 Display Segmentation Maps
                if "category_id_segmaps" in f:
                    seg = f["category_id_segmaps"]
                    seg_img = np.array(seg, dtype=np.uint8)
                    cv2.imshow(f"Segmentation Map - {hdf5_file} ", seg_img)
                    cv2.waitKey(0)

                # 🔹 Display Depth Maps
                if "depth" in f:
                    depth = f["depth"]
                    depth_img = np.array(depth).astype(np.uint16)
                    depth_img = (depth_img - np.min(depth_img)) / (np.max(depth_img) - np.min(depth_img) + 1e-6)  # Normalize
                    depth_img = (depth_img * 255).astype(np.uint8)
                    cv2.imshow(f"Depth Map - {hdf5_file} ", depth_img)
                    cv2.waitKey(0)

                # 🔹 Display Normal Maps
                if "normals" in f:
                    norm = f["normals"]
                    normal_img = ((np.array(norm) + 1) / 2 * 255).astype(np.uint8)  # Normalize to [0,255]
                    cv2.imshow(f"Normal Map - {hdf5_file} ", cv2.cvtColor(normal_img, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(0)

                # 🔹 Print Object Poses
                if "poses" in f:
                    print("\nObject Poses:")
                    pose = f["poses"]
                    print(f"Pose :\n", np.array(pose))
                    input("Press Enter to continue...")

                # 🔹 Print Bounding Boxes
                if "bounding_boxes" in f:
                    print("\nBounding Boxes:")
                    bbox = f["bounding_boxes"]
                    print(f"Bounding Box :", np.array(bbox))
                    input("Press Enter to continue...")

                # Close OpenCV windows
                cv2.destroyAllWindows()

if __name__ == "__main__":
    base_folder = "output"  # Change this to your actual base folder
    visualize_hdf5_files(base_folder)
