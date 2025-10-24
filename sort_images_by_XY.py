import os
import shutil
import re

def sort_images_by_XY(images_folder):
    # Path where your files are located

    # Regex to capture XY followed by digits anywhere in filename
    pattern = re.compile(r"XY\d+")

    for fname in os.listdir(images_folder):
        fpath = os.path.join(images_folder, fname)

        if not os.path.isfile(fpath):
            continue  # skip directories

        match = pattern.search(fname)
        if not match:
            print(f"Skipping {fname}, no XY# found")
            continue

        group_name = match.group(0)  # e.g., XY1, XY23

        # If the file has RGB in its name → put in XY#_RGB folder
        if "RGB" in fname.upper():
            group_name = f"RGB/{group_name}_RGB"

        dest_folder = os.path.join(images_folder, group_name)
        print(images_folder)
        os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, fname)
        shutil.move(fpath, dest_path)

        print(f"Moved {fname} to {dest_folder}")

if __name__ == "__main__":
    base_folder = r"C:\Users\billy\Documents\VIP Images\William_20250710_PDO device 1 to 8_for AI"

    for subfolder in sorted(os.listdir(base_folder)):
        subfolder_path = os.path.join(base_folder, subfolder)
        sort_images_by_XY(subfolder_path)