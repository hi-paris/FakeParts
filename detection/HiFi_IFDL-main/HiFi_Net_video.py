# ------------------------------------------------------------------------------
# Author: Samy AIMEUR greatly inspired by Xiao Guo (guoxia11@msu.edu)
# ------------------------------------------------------------------------------
from utils.utils import *
from utils.custom_loss import IsolatingLossFunction, load_center_radius_api
from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_cfg_defaults
from models.NLCDetection_api import NLCDetection
from utils.video_loader import RealFakeVideoDataset

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
import argparse
import imageio as imageio
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score


import json
import csv
import time
import argparse
import os


class HiFi_Net():
    '''
        FENET is the multi-branch feature extractor.
        SegNet contains the classification and localization modules.
        LOSS_MAP is the classification loss function class.
    '''
    def __init__(self):
        device = torch.device('cuda:0')
        device_ids = [0]

        FENet_cfg = get_cfg_defaults()
        FENet  = get_seg_model(FENet_cfg).to(device) # load the pre-trained model inside.
        SegNet = NLCDetection().to(device)
        FENet  = nn.DataParallel(FENet)
        SegNet = nn.DataParallel(SegNet)

        self.FENet  = restore_weight_helper(FENet,  "weights/HRNet",  750001)
        self.SegNet = restore_weight_helper(SegNet, "weights/NLCDetection", 750001)
        self.FENet.eval()
        self.SegNet.eval()

        center, radius = load_center_radius_api()
        self.LOSS_MAP = IsolatingLossFunction(center,radius).to(device)

    def _normalized_threshold(self, res, prob, threshold=0.5, verbose=False):
        '''to interpret detection result via omitting the detection decision.'''
        if res > threshold:
            decision = "Forged"
            prob = (prob - threshold) / threshold
        else:
            decision = 'Real'
            prob = (threshold - prob) / threshold
        print(f'Image being {decision} with the confidence {prob*100:.1f}.')

    def detect(self, video, verbose=False):
        """
            Para: image_name is string type variable for the image name.
            Return:
                res: binary result for real and forged.
                prob: the prob being the forged image.
        """
        with torch.no_grad():
            y_pred = []
            for i in range(len(video)):
                img_input = video[i:i+1]
                output = self.FENet(img_input)
                mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
                _, prob = one_hot_label_new(out3)
                y_pred.append(prob[0])
            return y_pred

    def localize(self, image):
        """
            Para: image_name is string type variable for the image name.
            Return:
                binary_mask: forgery mask.
        """
        with torch.no_grad():
            img_input = image
            output = self.FENet(img_input)
            mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, img_input)
            pred_mask, pred_mask_score = self.LOSS_MAP.inference(mask1_fea)   # inference
            pred_mask_score = pred_mask_score.cpu().numpy()
            ## 2.3 is the threshold used to seperate the real and fake pixels.
            ## 2.3 is the dist between center and pixel feature in the hyper-sphere.
            ## for center and pixel feature please refer to "IsolatingLossFunction" in custom_loss.py
            pred_mask_score[pred_mask_score<2.3] = 0.
            pred_mask_score[pred_mask_score>=2.3] = 1.
            binary_mask = pred_mask_score[0]
            ##ADDDD
            pred_mask = torch.zeros_like(mask1_binary)
            pred_mask[mask1_binary > 0.5] = 1
            pred_mask[mask1_binary <= 0.5] = 0
 
            return binary_mask


def inference(img_path):
    HiFi = HiFi_Net()   # initialize
    
    ## detection
    res3, prob3 = HiFi.detect(img_path)
    print(res3, prob3)
    HiFi.detect(img_path, verbose=True)
    
    ## localization
    path_mask = 'pred_mask.png'
    binary_mask = HiFi.localize(img_path)
    print("Type of binary mask :", binary_mask.dtype)
    print("Shape of binary mask :", binary_mask.shape)
    print("Sum of mask pixel :", binary_mask.sum())
    binary_mask = Image.fromarray((binary_mask*255.).astype(np.uint8))
    binary_mask.save(path_mask)
    print(f"Mask has been saved here : {path_mask}")


def validate(model, loader, dataset_name, result_folder, save_tensor=False, find_thres=False):

    with torch.no_grad():
        #y_true, y_pred = [], []
        print ("Number of batch: %d" %(len(loader)))
        print("Number of videos: ", len(loader.dataset))
        print("Batch size: ", loader.batch_size)
        #num = 0
        for i, (video_tensor, label) in enumerate(tqdm(loader, total=len(loader))):
            #print(f"Processing batch {i}")
            print("video_tensor shape: ", video_tensor.shape)

            #take the first element of the batch
            video = video_tensor[0]
            nb_frames = video.shape[0]

            video_path = loader.dataset.total_list[i]
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            in_tens = video.cuda()

            #y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            #y_true.extend(label.flatten().tolist())

            y_pred = model.detect(in_tens)
            y_true = [label[0]]*nb_frames

            y_true, y_pred = np.array(y_true), np.array(y_pred)

            #print(f"Shapes of y_true : {y_true.shape}, y_pred : {y_pred.shape}")
            #print(y_pred)
            #print(f"y_true: {y_true.shape}, y_pred: {y_pred.shape}")

            #saving tensor
            if save_tensor:
                dataset_saving_folder = os.path.join(result_folder, dataset_name)
                real_video_saving_dir = os.path.join(dataset_saving_folder, '0_real')
                fake_video_saving_dir = os.path.join(dataset_saving_folder, '1_fake')

                #Creating directories
                if not os.path.exists(dataset_saving_folder):
                    os.makedirs(dataset_saving_folder)
                if not os.path.exists(real_video_saving_dir):
                    os.makedirs(real_video_saving_dir)
                if not os.path.exists(fake_video_saving_dir):
                    os.makedirs(fake_video_saving_dir)

                if label[0] == 0:
                    tensor_path = os.path.join(real_video_saving_dir, video_name+'_results.pth')
                else:
                    tensor_path = os.path.join(fake_video_saving_dir, video_name+'_results.pth')

                torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ), tensor_path )
                print(f"Tensor has been saved at {tensor_path}")
    
    '''
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres
    '''

    return 1,1,1,1,1,1,1,1


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset_paths', type=str, default='scripts/dataset_path/test.json', help="Path to the json file containing the dataset paths")
parser.add_argument('-b','--batch_size', type=int, default=1)
parser.add_argument('-j','--workers', type=int, default=1, help='number of workers')
parser.add_argument('-s', '--save_tensor', default=None, action='store_true', help='save tensor')

parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--results_dir', type=str, default='results', help='directory to save results')

parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')
parser.add_argument('--shuffle', default=None, action='store_true', help='shuffle the dataset')

parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")

opt = parser.parse_args()

start_time = time.perf_counter()

# Setup path
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

# Create results directory
result_folder = opt.results_dir
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    

# Load model
model = HiFi_Net()

#Create csv results file
model_name = 'detection'
csv_name = opt.results_dir + '/{}.csv'.format(model_name)
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'real_acc', 'fake_acc', 'tot_accuracy', 'avg_precision']]
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)


# Dataset path
with open(opt.dataset_paths, 'r') as f:
    dataset_paths = json.load(f)

print("\n ===== dataset_paths ===== ")
print(dataset_paths)

for dataset_path in tqdm(dataset_paths, total=len(dataset_paths), desc='Number of datasets'):

        dataset_name = dataset_path['key']
        print('\nDataset: ', dataset_name)
        print("\ndataset_path: ", dataset_path) #AADDDDDDDDDDDD

        dataset = RealFakeVideoDataset(  dataset_path['real_path'], 
                                    dataset_path['fake_path'], 
                                    dataset_path['data_mode'], 
                                    opt.max_sample, 
                                    shuffle=opt.shuffle,
                                    jpeg_quality=opt.jpeg_quality, 
                                    gaussian_sigma=opt.gaussian_sigma,
                                    )

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, dataset_name, opt.results_dir, save_tensor=opt.save_tensor, find_thres=False)

        #saving results
        with open(csv_name, 'a') as f:
                csv_writer = csv.writer(f, delimiter=',')
                csv_writer.writerow([dataset_name, round(r_acc0, 4), round(f_acc0, 4), round(acc0, 4), round(ap, 4)])

torch.cuda.synchronize() # wait for GPU to finish
end_time = time.perf_counter()
inference_time = end_time - start_time  
print(f"CNNDetection inference time: {inference_time:.4f} s")