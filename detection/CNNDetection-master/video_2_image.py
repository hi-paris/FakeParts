import os
import shutil

# Define input and output directories
input_dir = "examples/realfakedir2/0_real"
output_dir = "examples/realfakedir3/0_real"

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Walk through subfolders in input_dir
for folder_name in os.listdir(input_dir):
    folder_path = os.path.join(input_dir, folder_name)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            if image_name.endswith(".png"):
                src_path = os.path.join(folder_path, image_name)
                dst_name = f"{folder_name}_{image_name}"
                dst_path = os.path.join(output_dir, dst_name)
                shutil.copy2(src_path, dst_path)
