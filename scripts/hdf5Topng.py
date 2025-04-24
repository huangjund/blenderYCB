import os
import h5py
import numpy as np
import cv2
import shutil

def save_hdf5_files(base_folder, output_folder):
    """
    Iterates through all 'rendered_scene_*' folders and visualizes HDF5 files using OpenCV, and saves
    RGB images as .jpg and depth maps as .png files.

    Parameters:
        base_folder (str): Path to the folder containing rendered scene folders.
        output_folder (str): Path to the folder where images will be saved.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all rendered scene folders and sort them numerically
    scene_folders = sorted(
        [f for f in os.listdir(base_folder) if f.startswith("rendered_scene_")],
        key=lambda x: int(x.split("_")[-1])  # Extract the numeric part and convert it to an integer
    )
    if not scene_folders:
        print("No 'rendered_scene_*' folders found.")
        return

    image_counter = 0  # For naming output images

    for scene_folder in scene_folders:
        scene_path = os.path.join(base_folder, scene_folder)
        print(f"\n🟢 Processing Scene: {scene_folder}")

        # Get all HDF5 files in the current scene folder and sort them numerically
        hdf5_files = sorted(
            [f for f in os.listdir(scene_path) if f.endswith(".hdf5")],
            key=lambda x: int(os.path.splitext(x)[0])  # Convert filename (without extension) to an integer
        )
        if not hdf5_files:
            print(f"⚠ No HDF5 files found in {scene_folder}. Skipping...")
            continue

        for hdf5_file in hdf5_files:
            file_path = os.path.join(scene_path, hdf5_file)
            print(f"\n🔹 Processing File: {hdf5_file}")

            with h5py.File(file_path, "r") as f:
                # 🔹 Save RGB Images as .jpg
                if "colors" in f:
                    img = f["colors"]
                    rgb_img = np.array(img).astype(np.uint8)
                    rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

                    # Save as .jpg
                    rgb_filename = os.path.join(output_folder, f"{image_counter:06d}-color.jpg")
                    cv2.imwrite(rgb_filename, rgb_img_bgr)
                    print(f"Saved: {rgb_filename}")

                # 🔹 Save Depth Maps as .png
                if "depth" in f:
                    depth = f["depth"]
                    depth_img = np.array(depth).astype(np.uint16)

                    # Save as .png
                    depth_filename = os.path.join(output_folder, f"{image_counter:06d}-depth.png")
                    cv2.imwrite(depth_filename, depth_img)
                    print(f"Saved: {depth_filename}")

                # 🔹 Save Depth Maps as .png
                if "category_id_segmaps" in f:
                    seg_img = f["category_id_segmaps"]
                    seg_img = np.array(seg_img).astype(np.uint8)

                    # Save as .png
                    seg_filename = os.path.join(output_folder, f"{image_counter:06d}-label.png")
                    cv2.imwrite(seg_filename, seg_img)
                    print(f"Saved: {seg_filename}")

                # copy meta.mat to the target directory
                meta_filename = os.path.join(output_folder, f"{image_counter:06d}-meta.mat")
                shutil.copy2(os.path.join(scene_path,f"{image_counter%len(hdf5_files)}_meta.mat"), meta_filename)

                # Increment image counter for next file
                image_counter += 1

    print("\n✅ All HDF5 files have been processed and saved.")

if __name__ == "__main__":
    base_folder = "output"  # Path to the folder containing 'rendered_scene_*'
    output_folder = "data_syn"  # Folder where images will be saved

    save_hdf5_files(base_folder, output_folder)
