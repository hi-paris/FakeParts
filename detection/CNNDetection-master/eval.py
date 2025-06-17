import os
import csv
import torch
import json
from tqdm import tqdm

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from util import mkdir

# === Load configuration from dataset.json ===
with open("dataset.json", "r") as f:
    dataset_config = json.load(f)[0]

real_path = dataset_config["real_path"]
fake_path = dataset_config["fake_path"]
dataset_key = dataset_config["key"]

# === Set result directory ===
results_dir = './results'
mkdir(results_dir)

# === Set model path (best model) ===
model_path = '/data/parietal/store3/work/gbrison/gen/detectors/other/Detectors/CNNDetection-master/checkpoints/blur_jpg_prob0.5/model_epoch_best.pth'
model_name = os.path.basename(model_path).replace('.pth', '')
csv_name = os.path.join(results_dir, f"{model_name}.csv")
per_image_csv = os.path.join(results_dir, f"{model_name}_per_image.csv")

# === Write global results CSV header ===
rows = [[f"{model_name} model testing on..."],
        ['testset', 'real_acc', 'fake_acc', 'tot_accuracy', 'avg precision']]
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(rows)

# === Setup options ===
opt = TestOptions().parse(print_options=False)
opt.real_path = real_path
opt.fake_path = fake_path
opt.results_dir = results_dir
opt.model_path = model_path
opt.no_resize = True
opt.classes = ['']  # binary classification

# === Patch for missing or capitalized attributes ===
if not hasattr(opt, 'load_size'):
    opt.load_size = getattr(opt, 'loadSize', 224)
if not hasattr(opt, 'batch_size'):
    opt.batch_size = getattr(opt, 'batchSize', 32)

# === Load model ===
model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
model.cuda()
model.eval()

# === Evaluate ===
print(f"Evaluating model on dataset: {dataset_key}")
acc, ap, r_acc, f_acc, y_true, y_pred, image_paths = validate(model, opt)

# === Output global metrics ===
print(f"({dataset_key}); r_acc: {r_acc}, f_acc: {f_acc}, acc: {acc}, ap: {ap}")
with open(csv_name, 'a') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow([dataset_key, round(r_acc, 4), round(f_acc, 4), round(acc, 4), round(ap, 4)])

# === Output per-image results ===
with open(per_image_csv, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['image_path', 'true_label', 'predicted_label', 'predicted_prob'])

    for i in range(len(y_true)):
        img_path = image_paths[i] if i < len(image_paths) else f"index_{i}"
        prob = y_pred[i]
        pred_label = 1 if prob >= 0.5 else 0
        writer.writerow([img_path, int(y_true[i]), pred_label, round(float(prob), 4)])

print(f"Per-image results saved to: {per_image_csv}")
print(f"The csv result file has been saved here : {csv_name}")
