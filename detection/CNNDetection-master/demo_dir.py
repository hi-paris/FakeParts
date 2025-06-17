import os
import argparse
import json
import csv
import time
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score

from networks.resnet import resnet50
from video_dataset.video_loader import RealFakeVideoDataset


def get_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > thres)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    ap = average_precision_score(y_true, y_pred)
    return ap, acc, r_acc, f_acc


def find_best_threshold(y_true, y_pred):
    best_acc, best_thres = 0, 0.5
    for thres in np.linspace(0.01, 0.99, 99):
        acc = accuracy_score(y_true, y_pred > thres)
        if acc > best_acc:
            best_acc = acc
            best_thres = thres
    return best_thres


def validate(model, loader, dataset_name, result_folder, save_tensor=False, find_thres=False):
    y_true_all, y_pred_all, path_all = [], [], []

    with torch.no_grad():
        print("Number of batches:", len(loader))
        print("Number of videos:", len(loader.dataset))
        print("Batch size:", loader.batch_size)

        for i, (video_tensor, label) in enumerate(tqdm(loader, total=len(loader))):
            video = video_tensor[0]  # (T, 3, H, W)
            nb_frames = video.shape[0]
            video_path = loader.dataset.total_list[i]
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            frame_preds = []
            frame_paths = []

            for j in range(nb_frames):
                frame = video[j].unsqueeze(0)  # (1, 3, H, W)
                if torch.cuda.is_available():
                    frame = frame.cuda()
                pred = torch.sigmoid(model(frame)).item()
                frame_preds.append(pred)

                frame_name = f"{video_name}_frame{j:03d}"
                frame_path = os.path.join(video_path, frame_name)  # Simulated full path
                frame_paths.append(frame_path)

            y_true = [label[0].item()] * nb_frames

            y_true_all.extend(y_true)
            y_pred_all.extend(frame_preds)
            path_all.extend(frame_paths)

            if save_tensor:
                dataset_saving_folder = os.path.join(result_folder, dataset_name)
                real_video_saving_dir = os.path.join(dataset_saving_folder, '0_real')
                fake_video_saving_dir = os.path.join(dataset_saving_folder, '1_fake')

                os.makedirs(real_video_saving_dir, exist_ok=True)
                os.makedirs(fake_video_saving_dir, exist_ok=True)

                save_dir = real_video_saving_dir if label[0] == 0 else fake_video_saving_dir
                tensor_path = os.path.join(save_dir, f"{video_name}_results.pth")
                torch.save(torch.stack([torch.tensor(y_true), torch.tensor(frame_preds)]), tensor_path)
                print(f"Tensor saved at {tensor_path}")

    # Save per-frame results
    csv_path = os.path.join(result_folder, f"{dataset_name}_per_frame.csv")
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_name', 'frame_path', 'predicted_prob', 'true_label'])
        for path, pred, label in zip(path_all, y_pred_all, y_true_all):
            writer.writerow([os.path.basename(path), path, round(pred, 6), int(label)])

    y_true_np = np.array(y_true_all)
    y_pred_np = np.array(y_pred_all)

    ap, acc0, r_acc0, f_acc0 = get_acc(y_true_np, y_pred_np, 0.5)

    if not find_thres:
        return ap, r_acc0, f_acc0, acc0, 0, 0, 0, 0

    best_thres = find_best_threshold(y_true_np, y_pred_np)
    _, acc1, r_acc1, f_acc1 = get_acc(y_true_np, y_pred_np, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres


# ========== Main script ==========

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_paths', type=str, required=True)
parser.add_argument('-m', '--model_path', type=str, required=True)
parser.add_argument('-b', '--batch_size', type=int, default=1)
parser.add_argument('-j', '--workers', type=int, default=4)
parser.add_argument('--use_cpu', action='store_true')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--results_dir', type=str, default='results')
parser.add_argument('--max_sample', type=int, default=1000)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--save_tensor', action='store_true')
parser.add_argument('--jpeg_quality', type=int, default=None)
parser.add_argument('--gaussian_sigma', type=int, default=None)

opt = parser.parse_args()

print("Starting inference...")
print(f"Using model: {opt.model_path}")
print(f"Dataset config: {opt.dataset_paths}")

# Setup
os.makedirs(opt.results_dir, exist_ok=True)

# Load model
model = resnet50(num_classes=1)
state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
model.eval()
if not opt.use_cpu:
    model.cuda()
print("Model loaded.")

# Image transform
trans_init = []
if opt.crop is not None:
    trans_init.append(transforms.CenterCrop(opt.crop))
else:
    trans_init.append(transforms.Resize((224, 224)))

trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset config
with open(opt.dataset_paths, 'r') as f:
    dataset_paths = json.load(f)

print("\n ===== dataset_paths ===== ")
print(dataset_paths)

# Process each dataset
for dataset_path in tqdm(dataset_paths, total=len(dataset_paths), desc="Number of datasets"):
    dataset_name = dataset_path['key']
    print(f"\nDataset: {dataset_name}")
    print(f"Real path: {dataset_path['real_path']}")
    print(f"Fake path: {dataset_path['fake_path']}")

    dataset = RealFakeVideoDataset(
        real_path=dataset_path['real_path'],
        fake_path=dataset_path['fake_path'],
        data_mode=dataset_path['data_mode'],
        max_sample=opt.max_sample,
        transform=trans,
        shuffle=opt.shuffle,
        jpeg_quality=opt.jpeg_quality,
        gaussian_sigma=opt.gaussian_sigma,
    )

    loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers)

    ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(
        model, loader, dataset_name, opt.results_dir, save_tensor=opt.save_tensor, find_thres=False)

    print(f"\nResults for {dataset_name}:")
    print(f"Real Acc: {r_acc0:.4f} | Fake Acc: {f_acc0:.4f} | Total Acc: {acc0:.4f} | AP: {ap:.4f}")
