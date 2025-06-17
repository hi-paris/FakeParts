import os
import csv
import torch
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm


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
            video = video_tensor[0]  # shape: (T, 3, H, W)
            nb_frames = video.shape[0]
            video_path = loader.dataset.total_list[i]
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            frame_preds = []

            for j in range(nb_frames):
                frame = video[j].unsqueeze(0)  # shape: (1, 3, H, W)
                if torch.cuda.is_available():
                    frame = frame.cuda()
                output = model(frame)
                prob = torch.sigmoid(output).item()
                frame_preds.append(prob)

            y_true = [label[0].item()] * nb_frames
            frame_names = [f"{video_name}_frame{j:03d}" for j in range(nb_frames)]

            y_true_all.extend(y_true)
            y_pred_all.extend(frame_preds)
            path_all.extend(frame_names)

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

    # Save per-frame results to CSV
    csv_path = os.path.join(result_folder, f"{dataset_name}_per_frame.csv")
    with open(csv_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_name', 'predicted_prob', 'true_label'])
        for path, pred, label in zip(path_all, y_pred_all, y_true_all):
            writer.writerow([path, round(pred, 6), int(label)])

    # Metrics
    y_true_np = np.array(y_true_all)
    y_pred_np = np.array(y_pred_all)

    ap, acc0, r_acc0, f_acc0 = get_acc(y_true_np, y_pred_np, 0.5)

    if not find_thres:
        return ap, r_acc0, f_acc0, acc0, 0, 0, 0, 0

    best_thres = find_best_threshold(y_true_np, y_pred_np)
    _, acc1, r_acc1, f_acc1 = get_acc(y_true_np, y_pred_np, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres
