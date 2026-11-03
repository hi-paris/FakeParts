from .clip import clip 
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.transformer_attention import TransformerAttention
import torchvision.transforms as transforms
from .clip.model import VisionTransformer
from .mlp import MLP

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-L/14-penultimate" : 1024
}


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

class CLIPModel(nn.Module):
    """UFD"""
    def __init__(self, name, num_classes=1):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )


    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)


class CLIPModelPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1):
        super(CLIPModelPenultimateLayer, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.register_hook()
        self.fc = nn.Linear(CHANNELS[name+"-penultimate"], num_classes)

    def register_hook(self):
        
        def hook(module, input, output):
            self.features = torch.clone(output)
        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)
        return 

    def forward(self, x):
        self.model.encode_image(x) 
        return self.fc(self.features)



class CLIPModelShuffleAttentionPenultimateLayer(nn.Module):
    def __init__(self, name, num_classes=1,shuffle_times=1, patch_size=32, original_times=1,
                 clip_download_root=None):
        super(CLIPModelShuffleAttentionPenultimateLayer, self).__init__()
        self.name = name
        self.num_classes = num_classes
        self.shuffle_times = shuffle_times
        self.original_times = original_times
        self.patch_size = patch_size
        self.model, self.preprocess = clip.load(
            name, device="cpu", download_root=clip_download_root
        ) # self.preprecess will not be used during training, which is handled in Dataset class
        self.register_hook()
        self.attention_head = TransformerAttention(CHANNELS[name+"-penultimate"], shuffle_times + original_times, last_dim=num_classes)
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def register_hook(self):
        
        def hook(module, input, output):
            self.features = torch.clone(output)
        for name, module in self.model.visual.named_children():
            if name == "ln_post":
                module.register_forward_hook(hook)
        return 
    
    def shuffle_patches(self, x, patch_size):
        B, C, H, W = x.size()
        # Unfold the input tensor to extract non-overlapping patches
        patches = F.unfold(x, kernel_size=patch_size, stride=patch_size, dilation=1)
        # Reshape the patches to (B, C, patch_H, patch_W, num_patches)
        shuffled_patches = patches[:, :, torch.randperm(patches.size(-1))]
        # Fold the shuffled patches back into images
        shuffled_images = F.fold(shuffled_patches, output_size=(H, W), kernel_size=patch_size, stride=patch_size)
        return shuffled_images


    def forward(self, x, return_feature=False):
        features = []
        with torch.no_grad():
            for i in range(self.shuffle_times):
                self.model.encode_image(self.shuffle_patches(x, patch_size=self.patch_size[0]))
                features.append(self.features)

            self.model.encode_image(x)
            for i in range(self.original_times):
                features.append(self.features.clone())
        features = self.attention_head(torch.stack(features, dim=-2))

        return features


