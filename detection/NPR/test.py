import sys
import time
import os
import csv
import torch
from util import Logger, printSet
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
import networks.resnet as resnet
import numpy as np
import random
import random
import argparse

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_torch(100)

DetectionTests = {
                'ForenSynths': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/ForenSynths/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

           'GANGen-Detection': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/GANGen-Detection/',
                                 'no_resize'  : True,
                                 'no_crop'    : True,
                               },

         'DiffusionForensics': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/DiffusionForensics/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

        'UniversalFakeDetect': { 'dataroot'   : '/opt/data/private/DeepfakeDetection/UniversalFakeDetect/',
                                 'no_resize'  : False, # Due to the different shapes of images in the dataset, resizing is required during batch detection.
                                 'no_crop'    : True,
                               },

                 }

def test_custom_dataset(opt, model, dataset_path, dataset_name="CustomDataset", no_resize=False, no_crop=True):
    """
    Test the model on a custom dataset folder.
    
    Args:
        opt: TestOptions object
        model: Loaded model
        dataset_path: Path to the dataset folder
        dataset_name: Name for the dataset (for display purposes)
        no_resize: Whether to skip resizing
        no_crop: Whether to skip cropping
    """
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist!")
        return
    
    printSet(dataset_name)
    print(f"Testing on dataset: {dataset_path}")
    
    # Check if the dataset has binary classification structure (0_real, 1_fake)
    subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    
    # Check if this is a binary classification dataset (has 0_real and 1_fake folders)
    has_real_fake_structure = all(folder in subfolders for folder in ['0_real', '1_fake'])
    
    if has_real_fake_structure:
        # This is a binary classification dataset, test it as a single dataset
        opt.dataroot = dataset_path
        opt.classes = ''
        opt.no_resize = no_resize
        opt.no_crop = no_crop
        
        try:
            acc, ap, _, _, _, _ = validate(model, opt)
            print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(0, dataset_name, acc*100, ap*100))
            print('*'*25)
        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")
    elif subfolders:
        # Dataset has other subfolders (like the existing structure with multiple categories)
        accs = []
        aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        
        for v_id, val in enumerate(subfolders):
            opt.dataroot = os.path.join(dataset_path, val)
            opt.classes = ''
            opt.no_resize = no_resize
            opt.no_crop = no_crop
            
            try:
                acc, ap, _, _, _, _ = validate(model, opt)
                accs.append(acc)
                aps.append(ap)
                print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
            except Exception as e:
                print(f"Error testing {val}: {e}")
                continue
        
        if accs:
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(len(accs), 'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100))
            print('*'*25)
    else:
        # Dataset is a single folder with mixed files, test directly
        opt.dataroot = dataset_path
        opt.classes = ''
        opt.no_resize = no_resize
        opt.no_crop = no_crop
        
        try:
            acc, ap, _, _, _, _ = validate(model, opt)
            print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(0, dataset_name, acc*100, ap*100))
            print('*'*25)
        except Exception as e:
            print(f"Error testing {dataset_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Test deepfake detection model')
    parser.add_argument('--model_path', required=True, help='Path to the model checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--custom_dataset', type=str, help='Path to custom dataset folder to test')
    parser.add_argument('--no_resize', action='store_true', help='Skip resizing for custom dataset')
    parser.add_argument('--no_crop', action='store_true', default=True, help='Skip cropping for custom dataset')
    parser.add_argument('--test_all', action='store_true', help='Test all predefined datasets')
    
    args = parser.parse_args()
    
    # Parse options
    opt = TestOptions().parse(print_options=False)
    opt.model_path = args.model_path
    opt.batch_size = args.batch_size
    
    print(f'Model_path {opt.model_path}')

    # get model
    model = resnet50(num_classes=1)
    model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
    model.cuda()
    model.eval()

    # Test custom dataset if specified
    if args.custom_dataset:
        test_custom_dataset(opt, model, args.custom_dataset, 
                           dataset_name="CustomDataset", 
                           no_resize=args.no_resize, 
                           no_crop=args.no_crop)
    
    # Test predefined datasets if requested or if no custom dataset specified
    if args.test_all or not args.custom_dataset:
        for testSet in DetectionTests.keys():
            dataroot = DetectionTests[testSet]['dataroot']
            printSet(testSet)

            accs = [];aps = []
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            for v_id, val in enumerate(os.listdir(dataroot)):
                opt.dataroot = '{}/{}'.format(dataroot, val)
                opt.classes  = '' #os.listdir(opt.dataroot) if multiclass[v_id] else ['']
                opt.no_resize = DetectionTests[testSet]['no_resize']
                opt.no_crop   = DetectionTests[testSet]['no_crop']
                acc, ap, _, _, _, _ = validate(model, opt)
                accs.append(acc);aps.append(ap)
                print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

if __name__ == "__main__":
    main() 

