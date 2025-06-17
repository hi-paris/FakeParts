import os
import sys
import time
import torch
import torch.nn
import argparse
import shutil
from PIL import Image
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.dataroot = '{}/{}/'.format(val_opt.dataroot, val_opt.val_split)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    return val_opt


def prepare_dataset_from_json(dataset_json_path):
    import json
    import tempfile

    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)

    entry = dataset[0]
    real_path = entry["real_path"]
    fake_path = entry["fake_path"]

    tmp_dir = tempfile.mkdtemp(prefix="cnn_data_")
    trainA_path = os.path.join(tmp_dir, 'trainA')
    trainB_path = os.path.join(tmp_dir, 'trainB')
    os.makedirs(trainA_path)
    os.makedirs(trainB_path)

    for fname in os.listdir(real_path):
        src = os.path.join(real_path, fname)
        dst = os.path.join(trainA_path, fname)
        os.symlink(os.path.abspath(src), dst)

    for fname in os.listdir(fake_path):
        src = os.path.join(fake_path, fname)
        dst = os.path.join(trainB_path, fname)
        os.symlink(os.path.abspath(src), dst)

    return tmp_dir


def prepare_val_split_from_json(dataset_json_path, num_val_images=20):
    import json
    import tempfile

    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)

    entry = dataset[0]
    real_path = entry["real_path"]
    fake_path = entry["fake_path"]

    tmp_val_dir = tempfile.mkdtemp(prefix="cnn_val_")
    valA_path = os.path.join(tmp_val_dir, 'valA')
    valB_path = os.path.join(tmp_val_dir, 'valB')
    os.makedirs(valA_path)
    os.makedirs(valB_path)

    for fname in sorted(os.listdir(real_path))[:num_val_images]:
        src = os.path.join(real_path, fname)
        dst = os.path.join(valA_path, fname)
        os.symlink(os.path.abspath(src), dst)

    for fname in sorted(os.listdir(fake_path))[:num_val_images]:
        src = os.path.join(fake_path, fname)
        dst = os.path.join(valB_path, fname)
        os.symlink(os.path.abspath(src), dst)

    return tmp_val_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_json', type=str, default=None, help='Path to dataset.json with real/fake paths')
    args, unknown = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + unknown
    opt = TrainOptions().parse()

    if args.dataset_json:
        opt.dataroot = prepare_dataset_from_json(args.dataset_json)
    else:
        opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)

    # Validation options
    if args.dataset_json:
        val_tmp_dir = prepare_val_split_from_json(args.dataset_json)
        val_opt = TrainOptions().parse(print_options=False)
        val_opt.dataroot = val_tmp_dir
        val_opt.isTrain = False
        val_opt.no_resize = False
        val_opt.no_crop = False
        val_opt.serial_batches = True
        val_opt.jpg_method = ['pil']
        if len(val_opt.blur_sig) == 2:
            b_sig = val_opt.blur_sig
            val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
        if len(val_opt.jpg_qual) != 1:
            j_qual = val_opt.jpg_qual
            val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]
    else:
        val_opt = get_val_opt()

    data_loader = create_dataloader(opt)
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)

    for epoch in range(opt.niter):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(data_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                train_writer.add_scalar('loss', model.loss, model.total_steps)

            if model.total_steps % opt.save_latest_freq == 0:
                print('saving the latest model %s (epoch %d, model.total_steps %d)' %
                      (opt.name, epoch, model.total_steps))
                model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, model.total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        # Validation
        model.eval()
        acc, ap = validate(model.model, val_opt)[:2]
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        early_stopping(acc, model)
        if early_stopping.early_stop:
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...")
                early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
            else:
                print("Early stopping.")
                break
        model.train()
