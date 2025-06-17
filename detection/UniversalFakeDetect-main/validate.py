import argparse
import os
import json
import csv
import time
import random
import pickle
from io import BytesIO
from copy import deepcopy

import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter

import supervision as sv
from models import get_model
from dataset_paths import DATASET_PATHS

# =============================================
# Global config
# =============================================
SEED = 0
MEAN = {
    "imagenet": [0.485, 0.456, 0.406],
    "clip":     [0.48145466, 0.4578275, 0.40821073]
}
STD = {
    "imagenet": [0.229, 0.224, 0.225],
    "clip":     [0.26862954, 0.26130258, 0.27577711]
}
# =============================================

def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def find_best_threshold(y_true, y_pred):
    """
    Find the threshold that maximizes overall accuracy.
    Assumes y_true has real (0) first then fake (1).
    """
    N = len(y_true)
    # perfectly separable
    if y_pred[:N//2].max() <= y_pred[N//2:].min():
        return (y_pred[:N//2].max() + y_pred[N//2:].min()) / 2

    best_acc = 0.0
    best_th = 0.0
    for th in y_pred:
        preds = (y_pred >= th).astype(int)
        acc = (preds == y_true).mean()
        if acc >= best_acc:
            best_acc = acc
            best_th = th
    return best_th


def png2jpg(img: Image.Image, quality: int) -> Image.Image:
    """
    Convert a PIL Image to JPEG at given quality and reload it.
    """
    out = BytesIO()
    img.save(out, format='JPEG', quality=quality)
    out.seek(0)
    jpg = Image.open(out)
    return jpg


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    """
    Apply Gaussian blur per channel.
    """
    arr = np.array(img)
    for c in range(arr.shape[2]):
        gaussian_filter(arr[:, :, c], sigma=sigma, output=arr[:, :, c])
    return Image.fromarray(arr)


def calculate_acc(y_true: np.ndarray, y_pred: np.ndarray, thres: float):
    """
    Returns (real_acc, fake_acc, overall_acc) at given threshold.
    """
    real_mask = y_true == 0
    fake_mask = y_true == 1
    r_acc = accuracy_score(y_true[real_mask], (y_pred[real_mask] > thres).astype(int))
    f_acc = accuracy_score(y_true[fake_mask], (y_pred[fake_mask] > thres).astype(int))
    tot_acc = accuracy_score(y_true, (y_pred > thres).astype(int))
    return r_acc, f_acc, tot_acc


def validate(model, loader, dataset_name, result_folder, save_tensor=False, find_thres=False):
    """
    Runs model on loader, writes per-frame scores CSV,
    and returns dataset-level metrics.
    """
    model.eval()
    batch_size = 4

    # Prepare image-level CSV
    img_csv_path = os.path.join(result_folder, f"{dataset_name}_image_scores.csv")
    img_csv = open(img_csv_path, 'w', newline='')
    img_writer = csv.writer(img_csv)
    img_writer.writerow(["image_path", "label", "score"])

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for idx, (video_tensor, label) in enumerate(tqdm(loader, desc=f"Validating {dataset_name}")):
            data_path = loader.dataset.total_list[idx]
            frames_tensor = video_tensor[0]  # [T,3,H,W]
            num_frames = frames_tensor.shape[0]

            # Predict in small batches to avoid OOM
            scores = []
            for j in range(0, num_frames, batch_size):
                batch = frames_tensor[j:j+batch_size].cuda()
                logits = model(batch).sigmoid().flatten().cpu().numpy()
                scores.extend(logits.tolist())

            y_true = np.array([label] * num_frames)
            y_pred = np.array(scores)

            # Write image-level rows
            if os.path.isdir(data_path):
                frame_files = sorted(os.listdir(data_path))
                for i_f, fname in enumerate(frame_files):
                    img_writer.writerow([os.path.join(data_path, fname), int(y_true[i_f]), float(y_pred[i_f])])
            else:
                for i_f in range(num_frames):
                    img_writer.writerow([f"{data_path}_frame{i_f:04d}", int(y_true[i_f]), float(y_pred[i_f])])

            all_y_true.extend(y_true.tolist())
            all_y_pred.extend(y_pred.tolist())

            # Optional: save raw tensors per video
            if save_tensor:
                save_dir = os.path.join(result_folder, dataset_name, '0_real' if label==0 else '1_fake')
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.basename(data_path).replace('/', '_') + '_results.pth'
                torch.save(torch.stack([torch.tensor(y_true), torch.tensor(y_pred)]),
                           os.path.join(save_dir, fname))

    img_csv.close()

    # Compute dataset-level metrics
    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    ap = average_precision_score(y_true_arr, y_pred_arr)
    r0, f0, t0 = calculate_acc(y_true_arr, y_pred_arr, 0.5)

    if not find_thres:
        return ap, r0, f0, t0, 0, 0, 0, 0

    best_th = find_best_threshold(y_true_arr, y_pred_arr)
    r1, f1, t1 = calculate_acc(y_true_arr, y_pred_arr, best_th)
    return ap, r0, f0, t0, r1, f1, t1, best_th


class RealFakeVideoDataset(Dataset):
    def __init__(self, real_path, fake_path, data_mode, max_sample,
                 arch, transforms_init, shuffle=False,
                 jpeg_quality=None, gaussian_sigma=None):
        assert data_mode in ['wang2020', 'ours']
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        # Gather lists
        self.real_list, self.fake_list = [], []
        def read_subset(rp, fp):
            real_list = get_list(rp, must_contain='0_real') if data_mode=='wang2020' else get_list(rp)
            fake_list = get_list(fp, must_contain='1_fake') if data_mode=='wang2020' else get_list(fp)
            if max_sample >= 0:
                random.shuffle(real_list);
                random.shuffle(fake_list);
                real_list = real_list[:max_sample]
                fake_list = fake_list[:max_sample]
            return real_list, fake_list

        if isinstance(real_path, str):
            self.real_list, self.fake_list = read_subset(real_path, fake_path)
        else:
            for rp, fp in zip(real_path, fake_path):
                rl, fl = read_subset(rp, fp)
                self.real_list += rl
                self.fake_list += fl

        self.total_list = self.real_list + self.fake_list
        self.labels = {p: 0 for p in self.real_list}
        self.labels.update({p: 1 for p in self.fake_list})

        # Build transform pipeline
        norm_from = 'imagenet' if arch.lower().startswith('imagenet') else 'clip'
        self.transform = transforms.Compose(
            transforms_init + [
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[norm_from], std=STD[norm_from])
            ]
        )

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        path = self.total_list[idx]
        label = self.labels[path]
        frames = []
        ext = path.split('.')[-1].lower()

        if ext in ['mp4', 'avi']:
            for frm in sv.get_video_frames_generator(path, stride=1, start=0, end=None):
                frames.append(Image.fromarray(frm[..., ::-1]).convert('RGB'))
        else:
            files = sorted(os.listdir(path))
            for fn in files:
                img = Image.open(os.path.join(path, fn)).convert('RGB')
                frames.append(img)

        if self.jpeg_quality is not None:
            frames = [png2jpg(f, self.jpeg_quality) for f in frames]
        if self.gaussian_sigma is not None:
            frames = [gaussian_blur(f, self.gaussian_sigma) for f in frames]

        vids = [self.transform(f) for f in frames]
        return torch.stack(vids), label


def recursively_read(rootdir, must_contain='', exts=None):
    if exts is None:
        exts = ['png','jpg','jpeg','bmp','mp4']
    out = []
    for r, d, files in os.walk(rootdir):
        for fn in files:
            if fn.split('.')[-1].lower() in exts and must_contain in fn:
                path = os.path.join(r, fn)
                # for images, return directory once
                if fn.split('.')[-1].lower() in ['png','jpg','jpeg','bmp']:
                    out.append(r)
                    break
                out.append(path)
    return out


def get_list(path, extensions=None, must_contain=''):
    if path.endswith('.pickle'):
        with open(path, 'rb') as f:
            items = pickle.load(f)
        return [i for i in items if must_contain in i]
    if extensions is None:
        extensions = ['png','jpg','jpeg','bmp','mp4']
    return recursively_read(path, must_contain, extensions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset_paths', type=str,
                        default='scripts/dataset_path/test.json',
                        help='Path to dataset JSON')
    parser.add_argument('--max_sample', type=int, default=1000)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('-c', '--crop', type=int, default=224)
    parser.add_argument('--resize', type=int, default=None)
    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')
    parser.add_argument('--result_folder', type=str, default='result')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--jpeg_quality', type=int, default=None)
    parser.add_argument('--gaussian_sigma', type=int, default=None)
    parser.add_argument('--save_tensor', action='store_true')
    opt = parser.parse_args()

    # Start
    start_time = time.perf_counter()
    os.makedirs(opt.result_folder, exist_ok=True)

    # Summary CSV
    with open(os.path.join(opt.result_folder, 'acc0.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'real_acc', 'fake_acc', 'tot_accuracy', 'avg_precision'])

    # Load model
    model = get_model(opt.arch)
    state = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state)
    model.cuda().eval()

    # Load dataset paths
    if opt.dataset_paths is None:
        dpaths = DATASET_PATHS
    else:
        with open(opt.dataset_paths, 'r') as f:
            dpaths = json.load(f)

    # Transform init
    trans_init = []
    if opt.resize is not None:
        trans_init.append(transforms.Resize(opt.resize))
    if opt.crop is not None:
        trans_init.append(transforms.CenterCrop(opt.crop))

    # Iterate datasets
    for dp in tqdm(dpaths, desc='Datasets'):
        set_seed()
        key = dp['key']
        dataset = RealFakeVideoDataset(
            dp['real_path'], dp['fake_path'], dp['data_mode'],
            opt.max_sample, opt.arch, trans_init,
            shuffle=opt.shuffle,
            jpeg_quality=opt.jpeg_quality,
            gaussian_sigma=opt.gaussian_sigma
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4
        )

        ap, r0, f0, t0, r1, f1, t1, bt = validate(
            model, loader, key, opt.result_folder,
            save_tensor=opt.save_tensor, find_thres=True
        )

        with open(os.path.join(opt.result_folder, 'acc0.csv'), 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([key, round(r0, 4), round(f0, 4), round(t0, 4), round(ap, 4)])

    torch.cuda.synchronize()
    print(f"All done in {time.perf_counter() - start_time:.2f} seconds.")
