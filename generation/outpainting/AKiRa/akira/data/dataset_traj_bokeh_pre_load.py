import os
import random
import json
import torch

import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

from PIL import Image
from torch.utils.data.dataset import Dataset
from packaging import version as pver
from akira.utils.ray_utils import Trajectory
from akira.utils.util import Args
from decord import VideoReader
from decord import cpu, gpu


class RandomHorizontalFlipWithPose(nn.Module):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlipWithPose, self).__init__()
        self.p = p

    def get_flip_flag(self, n_image):
        return torch.rand(n_image) < self.p

    def forward(self, image, flip_flag=None):
        n_image = image.shape[0]
        if flip_flag is not None:
            assert n_image == flip_flag.shape[0]
        else:
            flip_flag = self.get_flip_flag(n_image)

        ret_images = []
        for fflag, img in zip(flip_flag, image):
            if fflag:
                ret_images.append(F.hflip(img))
            else:
                ret_images.append(img)
        return torch.stack(ret_images, dim=0)


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

class RealEstate10KPose(Dataset):
    def __init__(
        self,
        root_path,
        annotation_json,
        sample_stride=4,
        minimum_sample_stride=1,
        sample_n_frames=16,
        relative_pose=False,
        zero_t_first_frame=False,
        sample_size=[256, 384],
        rescale_fxy=False,
        shuffle_frames=False,
        use_flip=False,
        return_clip_name=False,
        rand_focal_length=False,
        rand_fl_lbound=0.5,
        rand_fl_ubound=2.0,
        rand_focal_length_speed=0.1,
        rand_dist=False,
        rand_dist_bound=0.1,
        rand_dist_speed=0.01,
        rand_bokeh=False,
        rand_bokeh_fp_bound=0.3,
        rand_bokeh_fp_speed=0.01,
        rand_bokeh_K_bound=100,
        rand_bokeh_K_speed=2,
        plucker_type="cctrl",
        aug_dropout=0.2,
        static_aug_dropout=0.05,
        depth_folder="depth_clips",
        bokeh_reprez="power",
    ):
        self.root_path = root_path
        self.relative_pose = relative_pose
        self.zero_t_first_frame = zero_t_first_frame
        self.sample_stride = sample_stride
        self.minimum_sample_stride = minimum_sample_stride
        self.sample_n_frames = sample_n_frames
        self.return_clip_name = return_clip_name

        self.rand_dist = rand_dist
        self.rand_dist_bound = rand_dist_bound
        self.rand_dist_speed = rand_dist_speed

        self.rand_bokeh = rand_bokeh
        self.rand_bokeh_fp_bound = rand_bokeh_fp_bound
        self.rand_bokeh_fp_speed = rand_bokeh_fp_speed
        self.rand_bokeh_K_bound = rand_bokeh_K_bound
        self.rand_bokeh_K_speed = rand_bokeh_K_speed
        self.rand_focal_length = rand_focal_length
        self.rand_fl_lbound = rand_fl_lbound
        self.rand_fl_ubound = rand_fl_ubound 
        
        # plucker reprez
        self.plucker_type = plucker_type
        
        self.aug_dropout = aug_dropout
        self.static_aug_dropout = static_aug_dropout
        self.depth_folder = depth_folder

        self.dataset = json.load(open(os.path.join(root_path, annotation_json), "r"))
        self.length = len(self.dataset)
        self.rand_focal_length_speed = rand_focal_length_speed
        
        self.bokeh_reprez = bokeh_reprez

        sample_size = (
            tuple(sample_size)
            if not isinstance(sample_size, int)
            else (sample_size, sample_size)
        )
        self.sample_size = sample_size
        if use_flip: # TODO: remove this
            pixel_transforms = [
                transforms.Resize(sample_size),
                RandomHorizontalFlipWithPose(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        else:
            pixel_transforms = [
                transforms.Resize(sample_size),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        self.rescale_fxy = rescale_fxy
        self.sample_wh_ratio = sample_size[1] / sample_size[0]

        self.pixel_transforms = pixel_transforms
        self.shuffle_frames = shuffle_frames
        self.use_flip = use_flip

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if self.zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array(
            [[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [
            target_cam_c2w,
        ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def load_video_reader(self, idx):
        video_dict = self.dataset[idx]
        video_path = os.path.join(self.root_path, video_dict["clip_path"][0])
        video_reader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        return video_dict["clip_name"], video_reader, video_dict["caption"]

    def load_depth_imgs(self, idx, frame_indices=None):
        video_dict = self.dataset[idx]

        # in the folder structure, the depth clips are stored in the same folder as the clips
        depth_path = os.path.join(
            self.root_path,
            video_dict["clip_path"][0]
            .replace("clips", self.depth_folder)
            .replace(".mp4", ""),
        )
        if not os.path.isdir(depth_path):
            return []

        # get all the directory in this path
        depth_imgs = [
            os.path.join(depth_path, img) for img in sorted(os.listdir(depth_path))
        ]

        # select only the frame_indices
        if frame_indices is not None:
            if frame_indices.max() > len(depth_imgs):  # incomplete depth images
                depth_imgs = []  # return empty list
            else:
                depth_imgs = [depth_imgs[i] for i in frame_indices]

        return depth_imgs

    def load_cameras(self, idx):
        video_dict = self.dataset[idx]
        pose_file = os.path.join(self.root_path, video_dict["pose_file"])
        with open(pose_file, "r") as f:
            poses = f.readlines()
        poses = [pose.strip().split(" ") for pose in poses[1:]]
        cam_params = [[float(x) for x in pose] for pose in poses]
        cam_list = [Camera(cam_param) for cam_param in cam_params]
        return cam_list, poses

    def get_batch(self, idx):
        clip_name, video_reader, video_caption = self.load_video_reader(idx)
        cam_params, cam_params_raw = self.load_cameras(idx)
        assert len(cam_params) >= self.sample_n_frames
        total_frames = len(cam_params)

        current_sample_stride = self.sample_stride

        if total_frames < self.sample_n_frames * current_sample_stride:
            maximum_sample_stride = int(total_frames // self.sample_n_frames)
            current_sample_stride = random.randint(
                self.minimum_sample_stride, maximum_sample_stride
            )

        cropped_length = self.sample_n_frames * current_sample_stride
        start_frame_ind = random.randint(0, max(0, total_frames - cropped_length - 1))
        end_frame_ind = min(start_frame_ind + cropped_length, total_frames)

        assert end_frame_ind - start_frame_ind >= self.sample_n_frames
        frame_indices = np.linspace(
            start_frame_ind, end_frame_ind - 1, self.sample_n_frames, dtype=int
        )
        # load depth images for computing bokeh
        depth_imgs_dir = self.load_depth_imgs(idx, frame_indices)
        if len(depth_imgs_dir) == self.sample_n_frames:
            depth_imgs = [np.array(Image.open(img)) for img in depth_imgs_dir]
        else:
            depth_imgs = None
        
        if self.shuffle_frames:
            # Xi: do not do this ...
            # assert False, "Do not shuffle frames"
            # perm = np.random.permutation(self.sample_n_frames)
            # frame_indices = frame_indices[perm]
            pass

        # pixel read here
        pixel_values = (
            torch.from_numpy(video_reader.get_batch(frame_indices).asnumpy())
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        pixel_values = pixel_values / 255.0

        cam_params_raw = [cam_params_raw[indice] for indice in frame_indices]

        ori_h, ori_w = pixel_values.shape[-2:]
        dict_args = {
            "input": input,
            "image_width": self.sample_size[1],
            "image_height": self.sample_size[0],
            "video_length": len(cam_params_raw),
            "ori_width": ori_w,
            "ori_height": ori_h,
            "gpu_id": 0,
            "is_rectify_intrinsic": True,
            "get_relative_pose": self.relative_pose,
            "plucker_type": self.plucker_type
        }

        traj_args = Args(dict_args)
        traj = Trajectory()
        traj.read_from_camera_list(cam_params_raw, traj_args)
        # from zero
        if self.relative_pose:
            traj.apply_relative_pose()
        
        # this might be dangerous
        # # 5% of the time, we use static video    
        # self.use_static_video = np.random.uniform(0, 1) < self.static_aug_dropout
        # if self.use_static_video:
        #     # remove all poses
        #     traj.clear_all_pose()
        #     # static video
        #     pixel_values = pixel_values[0].repeat(16, 1, 1, 1)
        #     # static depth video
        #     if depth_imgs is not None:
        #         depth_imgs = [ depth_imgs[0] for _ in range(16) ]
        
        is_distortion_aug = False
        #print("augmentation")
        # probablity of augmentation
        # 5% of the time, we do not apply any augmentation
        if np.random.uniform(0, 1) > 0.05:
            # apply bokeh effect
            if depth_imgs is not None and self.rand_bokeh and np.random.uniform(0, 1) > self.aug_dropout:
                if len(depth_imgs) == self.sample_n_frames:
                    traj.rand_bokeh(
                        fp_bound=self.rand_bokeh_fp_bound,
                        fp_speed_rate=self.rand_bokeh_fp_speed,
                        K_bound=self.rand_bokeh_K_bound,
                        K_speed_rate=self.rand_bokeh_K_speed,
                    )
                #print("rand bokeh")
                
            # # apply distortion augmentation
            if self.rand_dist and np.random.uniform(0, 1) > self.aug_dropout:
                traj.rand_intrinsics_distortions(
                    -self.rand_dist_bound,
                    self.rand_dist_bound,
                    speed_rate=self.rand_dist_speed,
                )
                is_distortion_aug = True
                # print("rand dist")
                
            # focal length changed by distortion already!
            if self.rand_focal_length and np.random.uniform(0, 1) > self.aug_dropout and not is_distortion_aug:
                traj.rand_focal_length(
                    self.rand_fl_lbound,
                    self.rand_fl_ubound,
                    speed_rate=self.rand_focal_length_speed,
                )
                # print("rand focal")
        # print("augmentation end")

        # get the intrinsic after distortion (optional)
        plucker_embedding = (
            traj.get_plucker_embeddings_bokehs(use_bokeh=self.rand_bokeh, device="cpu", bokeh_reprez=self.bokeh_reprez).squeeze(0).contiguous()
        )

        # Xi: this is weird, we might shall never use flip flag, this is just for the sake of compatibility
        flip_flag = torch.zeros(self.sample_n_frames, dtype=torch.bool, device="cpu")
        return (
            pixel_values,
            video_caption,
            depth_imgs,
            plucker_embedding,
            flip_flag,
            clip_name,
            traj,
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                video, video_caption, depth_imgs, plucker_embedding, flip_flag, clip_name, traj = (
                    self.get_batch(idx)
                )
                break

            except Exception as e:
                print(f"Error loading data for index {idx}: {e}")
                idx = random.randint(0, self.length - 1)

        sample = dict(
            pixel_values=video,
            text=video_caption,
            depth_imgs=depth_imgs,
            plucker_embedding=plucker_embedding,
            clip_name=clip_name,
            traj=traj,
        )
        return sample

    @staticmethod
    def collate_fn(batch):
        plucker_embeddings = torch.stack([item["plucker_embedding"] for item in batch])
        # list
        pixel_values = [item["pixel_values"] for item in batch]
        depth_imgs = [item["depth_imgs"] for item in batch]
        video_captions = [item["text"] for item in batch]
        clip_names = [item["clip_name"] for item in batch]
        trajs = [item["traj"] for item in batch]

        return {
            "pixel_values": pixel_values,
            "text": video_captions,
            "depth_imgs": depth_imgs,
            "plucker_embedding": plucker_embeddings,
            "clip_names": clip_names,
            "trajs": trajs,
        }

    # data augmentation for training
    def get_pre_process_and_augmentation(self, batch, device="cuda", val=False):
        batch_depth_imgs = batch["depth_imgs"]
        batch_traj = batch["trajs"]
        _batch_pixel_values = batch["pixel_values"]
        batch_pixel_values = []
        for b in range(len(batch_traj)):
            traj = batch_traj[b]
            pixel_values = _batch_pixel_values[b]
            depth_imgs = batch_depth_imgs[b]

            # TODO: unify the size of the image ()
            w = traj.cameras[0].oc_info.w
            h = traj.cameras[0].oc_info.h

            # apply bokeh effect
            if depth_imgs is not None and self.rand_bokeh:
                if len(depth_imgs) == self.sample_n_frames:
                    pixel_values, _ = traj.bokeh_imgs(
                        pixel_values, depth_imgs, device=device
                    )

            # apply distortion augmentation
            if self.rand_dist:
                pixel_values = traj.distort_imgs(
                    pixel_values, device=device
                )
                
            # resize to plucker_embedding size
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )

            batch_pixel_values.append(pixel_values)
        
        batch_aug = batch  # necessary?
        # remove depth
        batch_aug.pop("depth_imgs")
        batch_pixel_values = torch.stack(batch_pixel_values, dim=0)
        
        # if val:
        batch_pixel_values = self.pixel_transforms[0](batch_pixel_values.squeeze(0)).unsqueeze(0)
        batch_pixel_values = self.pixel_transforms[1](batch_pixel_values)
        
        batch_aug["pixel_values"]  = batch_pixel_values
        
        # insert the first frame as condition image
        # condition_image_ind = random.sample(list(set(range(total_frames)) - set(frame_indices.tolist())), 1)
        # condition_image = torch.from_numpy(video_reader.get_batch(condition_image_ind).asnumpy()).permute(0, 3, 1, 2).contiguous()
        # condition_image = condition_image / 255.
        batch_aug["condition_image"] = batch_pixel_values[:, 0:1, :, :, :]

        batch_aug["video_caption"] = batch["text"]        
        return batch_aug
    
    @staticmethod
    def get_pre_process_and_augmentation_condition_image(batch, aug_dist, aug_bokeh=False, aug_idx=[0], device="cuda"):
          batch_depth_imgs = batch["depth_imgs"]
          batch_traj = batch["trajs"]
          _batch_pixel_values = batch["pixel_values"]
          batch_pixel_values = []
          for b in aug_idx:
              traj = batch_traj[b]
              pixel_values = _batch_pixel_values[b]
              depth_imgs = batch_depth_imgs[b]

              w = traj.cameras[0].oc_info.w
              h = traj.cameras[0].oc_info.h

              # apply bokeh effect
              if depth_imgs is not None and aug_bokeh:
                  pixel_values, _ = traj.bokeh_imgs(
                      pixel_values, depth_imgs, device=device
                  )

              # apply distortion augmentation
              if aug_dist:
                  pixel_values = traj.distort_imgs(
                      pixel_values, device=device
                  )
                  
              # resize to plucker_embedding size
              pixel_values = torch.nn.functional.interpolate(
                  pixel_values,
                  size=(h, w),
                  mode="bilinear",
                  align_corners=False,
              )

              batch_pixel_values.append(pixel_values)
          return batch_pixel_values
