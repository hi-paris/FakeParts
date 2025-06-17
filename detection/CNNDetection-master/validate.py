import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score, accuracy_score

from networks.resnet import resnet50
from options.test_options import TestOptions
from data.custom_dataset import CustomRealFakeDataset
from torch.utils.data import DataLoader


def validate(model, opt):
    # Create dataset directly from real/fake paths
    dataset = CustomRealFakeDataset(opt.real_path, opt.fake_path, opt)
    data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)

    with torch.no_grad():
        y_true, y_pred, image_paths = [], [], []
        print("Number of batches:", len(data_loader))
        print("Number of images:", len(data_loader.dataset))
        print("Batch size:", data_loader.batch_size)

        for img, label, paths in tqdm(data_loader, desc="Inference on dataset"):
            in_tens = img.cuda()
            logits = model(in_tens)
            probs = logits.sigmoid().flatten().tolist()
            y_pred.extend(probs)
            y_true.extend(label.flatten().tolist())
            image_paths.extend(paths)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)
    f_acc = accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return acc, ap, r_acc, f_acc, y_true, y_pred, image_paths


if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    # Set necessary params if missing
    opt.load_size = getattr(opt, 'load_size', 224)
    opt.batch_size = getattr(opt, 'batch_size', 32)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred, image_paths = validate(model, opt)

    print("Accuracy:", acc)
    print("Average Precision:", avg_precision)
    print("Accuracy of real images:", r_acc)
    print("Accuracy of fake images:", f_acc)
