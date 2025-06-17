import os
import csv

def sorted_image_list(folder, image_exts=('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
    """Return a sorted list of image file paths in the given folder."""
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(image_exts)
    ])

def split_csv_by_folder(csv_path, real_folder, fake_folder, output_real_csv, output_fake_csv, has_header=True):
    """Split a CSV into real and fake subsets based on folder image order."""
    # Get image files sorted by name
    real_images = sorted_image_list(real_folder)
    fake_images = sorted_image_list(fake_folder)

    # Count images
    n_real = len(real_images)
    n_fake = len(fake_images)

    # Read CSV rows
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.reader(f))
    
    header = reader[0] if has_header else None
    rows = reader[1:] if has_header else reader

    if len(rows) != (n_real + n_fake):
        print(f"Mismatch: {len(rows)} rows vs {n_real} real + {n_fake} fake images")
        return

    # Split and attach image paths
    real_rows = rows[:n_real]
    fake_rows = rows[n_real:n_real + n_fake]

    real_rows_with_path = [[real_images[i]] + real_rows[i] for i in range(n_real)]
    fake_rows_with_path = [[fake_images[i]] + fake_rows[i] for i in range(n_fake)]

    # Write real.csv
    with open(output_real_csv, 'w', newline='', encoding='utf-8') as f_real:
        writer = csv.writer(f_real)
        if has_header:
            writer.writerow(['image_path'] + header)
        writer.writerows(real_rows_with_path)

    # Write fake.csv
    with open(output_fake_csv, 'w', newline='', encoding='utf-8') as f_fake:
        writer = csv.writer(f_fake)
        if has_header:
            writer.writerow(['image_path'] + header)
        writer.writerows(fake_rows_with_path)

    print(f"✅ Done. Written {output_real_csv} and {output_fake_csv}")

# === Example usage ===
csv_path = '/data/parietal/store3/work/gbrison/gen/detectors/other/Detectors/CNNDetection-master/results/test_sample_per_frame.csv'
real_folder = '/data/parietal/store3/work/gbrison/gen/detectors/other/Detectors/CNNDetection-master/examples/realfakedir3/0_real'
fake_folder = '/data/parietal/store3/work/gbrison/gen/detectors/other/Detectors/CNNDetection-master/examples/realfakedir3/1_fake'
output_real_csv = 'real.csv'
output_fake_csv = 'fake.csv'

split_csv_by_folder(csv_path, real_folder, fake_folder, output_real_csv, output_fake_csv, has_header=True)
