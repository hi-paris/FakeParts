import os
import yaml
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import libs.eps_ad.utils as eps_ad_utils
from libs.eps_ad.runners.diffpure_ddpm import Diffusion
from libs.eps_ad.runners.diffpure_guided import GuidedDiffusion
from libs.eps_ad.runners.diffpure_sde import RevGuidedDiffusion
from libs.eps_ad.runners.diffpure_ode import OdeGuidedDiffusion
from libs.eps_ad.runners.diffpure_ldsde import LDGuidedDiffusion

def normalize_name(name):
    """
    e.g., ViT-B/32 -> ViT_B_32
    """
    parts = name.replace("/", "_").replace("-", "_").split()
    return "".join(part.upper() for part in parts)

def get_generation_models(dataset_name):
    if dataset_name == "GVF":
        return {
            "fake": {
                "train": ["GEN2", "keling", "ModelScopeT2V", "pika", "Show1", "sora", "T2V", "veo", "zeroscope"],
                "test": []
                },
            "real": {
                "train": ["MSR-VTT"],
                "test": []
            }
        }
    elif dataset_name == "GenVideo":
        return {
            "fake": {
                "train": [
                    "Pika", 
                    "SEINE", 
                ],
                "test": [
                    "ModelScope", 
                    "MorphStudio",  
                    "MoonValley", 
                    "HotShot",
                    "Show_1",
                    "Gen2", 
                    "Crafter",
                    "Lavie", 
                    "Sora", 
                    "WildScrape"
                ],
                "val": [
                    "ModelScope", 
                    "MorphStudio",  
                    "MoonValley", 
                    "HotShot",
                    "Show_1",
                    "Gen2", 
                    "Crafter",
                    "Lavie", 
                    "WildScrape"
                ]
            },
            "real": {
                "train": [
                    "Kinetics-400", 
                    "Youku-mPLUG"
                ],
                "test": [
                    "MSR-VTT"
                ]
            }
        }
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")

def get_all_generation_models(dataset_name):
    generation_models = get_generation_models(dataset_name)
    return [
        model 
        for category in ["fake", "real"] 
        for split in ["train", "test"] 
        for model in generation_models[category][split]
    ]

def get_label_from_generation_model(generation_model):
    if generation_model in ["MSVD", "MSR-VTT", "Kinetics-400", "Youku-mPLUG"]:
        return "real"
    else:
        return "fake"

class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config, process_shape=(3,256,256)):
        super().__init__()
        self.args = args

        # diffusion model
        print(f'diffusion_type: {self.args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(self.args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(self.args, config, device=config.device, process_shape=process_shape)
        elif args.diffusion_type == 'ode':
            self.runner = OdeGuidedDiffusion(self.args, config, device=config.device)
        elif args.diffusion_type == 'ldsde':
            self.runner = LDGuidedDiffusion(self.args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(self.args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.process_shape = process_shape
        self.tag = None

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x_re , ts_cat= self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag, t_size=self.args.t_size) # diffusion+purify

        x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)
        self.counter += 1

        return x_re, ts_cat

def parse_args_and_config(args_path, config_path):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args_path = os.path.abspath(args_path) if os.path.isabs(args_path) else os.path.join(repo_root, args_path)
    config_path = os.path.abspath(config_path) if os.path.isabs(config_path) else os.path.join(repo_root, config_path)
    def load_config(config_path):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def dict_to_args(config_dict):
        """Convert a dictionary to an argparse.Namespace object."""
        parser = argparse.ArgumentParser()
        for key, value in config_dict.items():
            parser.add_argument(f'--{key}', default=value, type=type(value))
        args = parser.parse_args([])
        return args
    
    config_path = args_path
    # Load configuration from YAML
    config = load_config(config_path)
    # Convert dictionary to argparse.Namespace
    args = dict_to_args(config)
    args.step_size_adv = args.epsilon / args.num_steps
    score_config_path = args.config
    if not os.path.isabs(score_config_path):
        score_config_path = os.path.join(repo_root, score_config_path)
    with open(score_config_path, 'r') as f:
        config = yaml.safe_load(f)
    new_config = eps_ad_utils.dict2namespace(config)
    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device
    return args, new_config

def get_score_fn(device='cuda',args_path='./libs/eps_ad/args.yml', config_path='./libs/eps_ad/imagenet.yml', process_shape=(3,256,256), diffusion_checkpoint=None):
    """Input should be: shape (batch_size, 3, 256, 256), value range [-1, 1]."""
    kargs, config = parse_args_and_config(args_path, config_path)
    kargs.diffusion_checkpoint = diffusion_checkpoint
    from libs.eps_ad.score_sde import sde_lib
    from libs.eps_ad.score_sde.models import utils as mutils
    config.device = device
    model = SDE_Adv_Model(kargs, config, process_shape)         
    model = model.eval().to(config.device)
    rev_vpsde = model.runner.rev_vpsde
    sde = sde_lib.VPSDE(beta_min=rev_vpsde.beta_0, beta_max=rev_vpsde.beta_1, N=rev_vpsde.N)
    score_fn = mutils.get_score_fn(sde, rev_vpsde.model, train=False, continuous=True)
    return score_fn
   
if __name__ == "__main__":
    name = "ViT-B/32"
    print(normalize_name(name))
