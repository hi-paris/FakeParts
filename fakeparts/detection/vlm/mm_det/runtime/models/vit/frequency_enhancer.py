import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur


class UpsampleProcessor:
    def __init__(self, scales=[4]):
        self.scales = scales
    
    def __call__(self, image):
        recons = list()
        for scale in self.scales:
            down_scale = 1 / scale
            down = F.interpolate(image, scale_factor=down_scale, mode='bilinear')
            up = F.interpolate(down, scale_factor=scale, mode='bilinear')
            recons.append(up)
        return torch.cat(recons, dim=1)
    

class GaussianBlurProcessor:
    def __init__(self, kernel_sizes=[3,5,7], sigmas=[None, None, None]):
        assert(len(kernel_sizes) == len(sigmas))
        self.kernel_sizes = kernel_sizes
        self.sigmas = sigmas
        self.blur_transforms = self.get_blur_transforms(self.kernel_sizes, self.sigmas)
    
    def get_blur_transforms(self, kernel_sizes, sigmas):
        blur_transforms = list()
        for kernel_size, sigma in zip(kernel_sizes, sigmas):
            if sigma is None:
                sigma = max(0.1, ((kernel_size - 1) * 0.5  - 1) * 0.5 + 0.8)
            blur_transforms.append(GaussianBlur(kernel_size, sigma))
        return blur_transforms
                
    def __call__(self, x):
        recons = list()
        for blur_transform in self.blur_transforms:
            blurred = blur_transform(x)
            recons.append(blurred)
        return torch.cat(recons, dim=1)


class ResidualGenerator:
    def __init__(self):
        pass
    
    def __call__(self, x, recon_processor):
        y = recon_processor(x)
        z = x.repeat(1, y.shape[1] // x.shape[1], 1, 1)
        return y - z