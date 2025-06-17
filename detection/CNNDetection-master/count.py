import os
import csv

def count_csv_lines(csv_path, include_header=False):
    """Count lines in a CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = sum(1 for _ in reader)
    return lines if include_header else max(0, lines - 1)

def count_lines_in_folder(folder_path, file_exts=('.csv', '.txt'), recursive=True):
    """Count lines in all text/CSV files in a folder."""
    total_lines = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(file_exts):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        file_lines = sum(1 for _ in f)
                        total_lines += file_lines
                        print(f"{path}: {file_lines} lines")
                except Exception as e:
                    print(f"Error reading {path}: {e}")
        if not recursive:
            break
    return total_lines

def count_image_files(folder_path, image_exts=('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'), recursive=True):
    """Count image files in a folder."""
    image_count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(image_exts):
                image_count += 1
        if not recursive:
            break
    return image_count

# === Example usage ===

csv_file = '/data/parietal/store3/work/gbrison/gen/detectors/other/Detectors/CNNDetection-master/results/test_sample_per_frame.csv'
folder_path = '/data/parietal/store3/work/gbrison/gen/detectors/other/Detectors/CNNDetection-master/examples/realfakedir3/1_fake'

csv_line_count = count_csv_lines(csv_file, include_header=False)
folder_line_count = count_lines_in_folder(folder_path, recursive=True)
image_file_count = count_image_files(folder_path, recursive=True)

print(f"\nCSV file lines (excluding header): {csv_line_count}")
print(f"Total text/CSV lines in folder: {folder_line_count}")
print(f"Total image files in folder: {image_file_count}")
