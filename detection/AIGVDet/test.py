#!/usr/bin/env python3
import argparse
import glob
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from core.utils1.utils import get_network, str2bool
from sklearn.metrics import average_precision_score, roc_auc_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-fop", "--folder_optical_flow_path", default="data/test/T2V/videocraft",
                        type=str, help="path to optical‐flow image folder")
    parser.add_argument("-for", "--folder_original_path", default="data/test/original/T2V/videocraft",
                        type=str, help="path to RGB/original image folder")
    parser.add_argument("-mop", "--model_optical_flow_path", type=str, default="checkpoints/optical.pth")
    parser.add_argument("-mor", "--model_original_path", type=str, default="checkpoints/original.pth")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="decision threshold")
    parser.add_argument("-e", "--excel_path", type=str, default="data/results/moonvalley_wang.csv",
                        help="path to CSV of video‐level results")
    parser.add_argument("-ef", "--excel_frame_path", type=str, default="data/results/frame/moonvalley_wang.csv",
                        help="path to CSV of frame‐level results")
    parser.add_argument("--use_cpu", action="store_true", help="force CPU even if CUDA is available")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--aug_norm", type=str2bool, default=True)
    args = parser.parse_args()

    # Load models
    model_op = get_network(args.arch)
    sd = torch.load(args.model_optical_flow_path, map_location="cpu")
    if "model" in sd: sd = sd["model"]
    model_op.load_state_dict(sd)
    model_op.eval()
    if not args.use_cpu:
        model_op.cuda()

    model_or = get_network(args.arch)
    sd = torch.load(args.model_original_path, map_location="cpu")
    if "model" in sd: sd = sd["model"]
    model_or.load_state_dict(sd)
    model_or.eval()
    if not args.use_cpu:
        model_or.cuda()

    trans = transforms.Compose([
        transforms.CenterCrop((448, 448)),
        transforms.ToTensor(),
    ])

    df_vid = pd.DataFrame(columns=['name', 'pro', 'flag', 'optical_pro', 'original_pro'])
    df_frame = pd.DataFrame(columns=['original_path', 'original_pro', 'optical_path', 'optical_pro', 'flag'])

    y_true, y_pred = [], []

    print("\n" + "*" * 50 + "\n")

    # Process folders
    for subfolder_name in ["0_real", "1_fake"]:
        flag = 0 if subfolder_name == "0_real" else 1
        rgb_root = os.path.join(args.folder_original_path, subfolder_name)
        flow_root = os.path.join(args.folder_optical_flow_path, subfolder_name)

        if not os.path.isdir(rgb_root):
            print("Skipping missing folder:", rgb_root)
            continue

        print("Processing class:", subfolder_name)
        for vid in sorted(os.listdir(rgb_root)):
            rgb_folder = os.path.join(rgb_root, vid)
            flow_folder = os.path.join(flow_root, vid)

            if not os.path.isdir(flow_folder):
                print("  missing flow folder:", flow_folder)
                continue

            print("  video:", vid)
            orig_files = sorted(glob.glob(os.path.join(rgb_folder, "*.jpg")) +
                                glob.glob(os.path.join(rgb_folder, "*.png")) +
                                glob.glob(os.path.join(rgb_folder, "*.JPEG")))
            flow_files = sorted(glob.glob(os.path.join(flow_folder, "*.jpg")) +
                                glob.glob(os.path.join(flow_folder, "*.png")) +
                                glob.glob(os.path.join(flow_folder, "*.JPEG")))

            if len(orig_files) == 0:
                print("    no frames, skipping")
                continue

            if len(orig_files) != len(flow_files):
                raise ValueError(f"frame‐count mismatch in {vid}: {len(orig_files)} RGB vs {len(flow_files)} flow")

            sum_or, sum_op = 0.0, 0.0

            for rgb_path, flow_path in tqdm(zip(orig_files, flow_files), total=len(orig_files),
                                            dynamic_ncols=True, desc=f"    frames"):
                # original
                img = Image.open(rgb_path).convert("RGB")
                tens = trans(img)
                if args.aug_norm:
                    tens = TF.normalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                batch = tens.unsqueeze(0)
                if not args.use_cpu:
                    batch = batch.cuda()
                with torch.no_grad():
                    p_or = model_or(batch).sigmoid().item()
                sum_or += p_or

                # optical flow
                img2 = Image.open(flow_path).convert("RGB")
                tens2 = trans(img2)
                if args.aug_norm:
                    tens2 = TF.normalize(tens2, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                batch2 = tens2.unsqueeze(0)
                if not args.use_cpu:
                    batch2 = batch2.cuda()
                with torch.no_grad():
                    p_op = model_op(batch2).sigmoid().item()
                sum_op += p_op

                df_frame.loc[len(df_frame)] = [rgb_path, p_or, flow_path, p_op, flag]

            mean_or = sum_or / len(orig_files)
            mean_op = sum_op / len(flow_files)
            vid_score = 0.5 * mean_or + 0.5 * mean_op

            print(f"    -> avg original={mean_or:.3f}, avg flow={mean_op:.3f}, final={vid_score:.3f}")

            y_true.append(flag)
            y_pred.append(vid_score)
            df_vid.loc[len(df_vid)] = [vid, vid_score, flag, mean_op, mean_or]

    # Compute statistics
    tp = sum((yt == 1) and (yp >= args.threshold) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0) and (yp < args.threshold) for yt, yp in zip(y_true, y_pred))
    p = sum(yt == 1 for yt in y_true)
    n = sum(yt == 0 for yt in y_true)

    print("\nRESULTS:")
    if n == 0:
        print("  TNR (true-neg rate): undefined (no negative samples)")
    else:
        print(f"  TNR (true-neg rate): {tn}/{n} = {tn/n:.3f}")

    if p == 0:
        print("  TPR (true-pos rate): undefined (no positive samples)")
    else:
        print(f"  TPR (true-pos rate): {tp}/{p} = {tp/p:.3f}")

    if (p + n) == 0:
        print("  ACC: undefined (no samples)")
    else:
        print(f"  ACC: {(tp + tn)}/{p + n} = {(tp + tn) / (p + n):.3f}")

    try:
        ap = average_precision_score(y_true, y_pred)
        print(f"  AP:  {ap:.3f}")
    except ValueError as e:
        print(f"  AP:  undefined (reason: {e})")

    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"  AUC: {auc:.3f}")
    except ValueError as e:
        print(f"  AUC: undefined (reason: {e})")

    print(f"\nProcessed {len(y_true)} videos total.")
    print(f"  -> Real videos: {n}")
    print(f"  -> Fake videos: {p}")

    os.makedirs(os.path.dirname(args.excel_path), exist_ok=True)
    df_vid.to_csv(args.excel_path, index=False)
    print("Wrote video‐level results to", args.excel_path)

    os.makedirs(os.path.dirname(args.excel_frame_path), exist_ok=True)
    df_frame.to_csv(args.excel_frame_path, index=False)
    print("Wrote frame‐level results to", args.excel_frame_path)
