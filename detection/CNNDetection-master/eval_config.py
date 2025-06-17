from util import mkdir
import json

# Directory to store the results
results_dir = './results'
mkdir(results_dir)

# Use your dataset defined in dataset.json
with open('dataset.json', 'r') as f:
    dataset_config = json.load(f)[0]

# Define your test root from dataset_config
real_path = dataset_config['real_path']
fake_path = dataset_config['fake_path']
dataroot = f"{real_path},{fake_path}"  # assuming your validate function can handle comma-separated paths

# Single dataset key (used for labeling in eval.py)
vals = [dataset_config['key']]

# Assume binary classification (real vs fake), so no multiclass
multiclass = [0]

# Update to point to your best model
model_path = '/data/parietal/store3/work/gbrison/gen/detectors/other/Detectors/CNNDetection-master/checkpoints/blur_jpg_prob0.5/model_epoch_best.pth'
