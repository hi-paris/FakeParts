import pandas as pd
import os
import glob

# Paths to the CSV and image folders
csv_path = "/data/parietal/store3/work/gbrison/gen/detectors/other/Detectors/CNNDetection-master/results/test_sample_per_frame.csv"  # Update this to your actual CSV path
real_dir = "/examples/realfakedir3/0_real"
fake_dir = "/examples/realfakedir3/0_fake"

# Load the CSV
df = pd.read_csv(csv_path)

# Get sorted list of image paths
real_images = sorted(glob.glob(os.path.join(real_dir, "*.png")))
fake_images = sorted(glob.glob(os.path.join(fake_dir, "*.png")))

# Indexes for tracking the next real/fake image to assign
real_idx = 0
fake_idx = 0

# List to store assigned image paths
assigned_paths = []

# Iterate over each row and assign image path based on label
for _, row in df.iterrows():
    if row['true_label'] == 0:
        assigned_paths.append(real_images[real_idx])
        real_idx += 1
    else:
        assigned_paths.append(fake_images[fake_idx])
        fake_idx += 1

# Add the new column to the dataframe
df['actual_image_path'] = assigned_paths

# Save the result to a new CSV
output_csv_path = "test_sample_with_image_paths.csv"  # Update as needed
df.to_csv(output_csv_path, index=False)

print(f"Updated CSV saved to: {output_csv_path}")
