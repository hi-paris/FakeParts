import os
import supervision as sv
import torch
import torch.utils.data
import numpy as np
from torch.utils.data import Dataset
from PIL import Image 
import imageio as imageio
import pickle
import random

from .utils import png2jpg
from .utils import gaussian_blur


def recursively_read(rootdir, must_contain="", exts=["png", "jpg", "JPEG", "jpeg", "bmp", "mp4"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            ext = file.split('.')[-1]
            if (ext in exts)  and  (must_contain in os.path.join(r, file)):
                if ext in ["png", "jpg", "JPEG", "jpeg", "bmp"]:
                    out.append(r)
                    break
                out.append(os.path.join(r, file))
                #print("Add video: ", os.path.join(r, file))
    return out

def get_list(path, extensions = None, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            video_list = pickle.load(f)
        video_list = [ item for item in video_list if must_contain in item   ]
    else:
        if extensions is None:
            extensions = ["png", "jpg", "JPEG", "jpeg", "bmp", "mp4"]
        video_list = recursively_read(path, must_contain, exts=extensions)
    return video_list


class RealFakeVideoDataset(Dataset):
    def __init__(self, real_path, fake_path, data_mode, max_sample, shuffle = None, jpeg_quality=None, gaussian_sigma=None):
        assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma

        # = = = = = = data path = = = = = = = = = #
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample, shuffle)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample, shuffle)
                real_list += real_l
                fake_list += fake_l
        
        # Store video paths and corresponding labels
        self.total_list = real_list + fake_list
        self.labels_dict = {path: 0 for path in real_list}
        self.labels_dict.update({path: 1 for path in fake_list})

    def read_path(self, real_path, fake_path, data_mode, max_sample, shuffle):
        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = get_list(fake_path)

        if max_sample is not None:
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[:max_sample]
            fake_list = fake_list[:max_sample]

        return real_list, fake_list

    def _transform_image(self, image):
        '''transform the image.'''
        image = image.resize((256,256), resample=Image.BICUBIC)
        image = np.asarray(image)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        return image

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        path = self.total_list[idx]
        label = self.labels_dict[path]
        frames = []

        if path.split('.')[-1] in ["mp4", "avi", "mkv"]:
            video_path = path
            # Load video frames using supervision
            frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)

            for frame in frame_generator:
                frame = Image.fromarray(frame[...,::-1])
                frames.append(frame)
        
        elif os.path.isdir(path):
            list_path_frames = os.listdir(path)
            list_path_frames.sort()

            for path_frames in list_path_frames:
                frame = Image.open(os.path.join(path,path_frames)).convert("RGB")
                frames.append(frame)

        if self.gaussian_sigma is not None:
            frames = [gaussian_blur(frame, self.gaussian_sigma) for frame in frames]
        if self.jpeg_quality is not None:
            frames = [png2jpg(frame, self.jpeg_quality) for frame in frames]

        # Apply transformations to each frame
        transformed_frames = [self._transform_image(frame) for frame in frames]

        # Stack frames to create a tensor
        video_tensor = torch.stack(transformed_frames)

        return video_tensor, label
