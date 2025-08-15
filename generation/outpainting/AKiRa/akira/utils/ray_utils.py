import torch
from packaging import version as pver
from scipy.interpolate import CubicSpline

# from wand.image import Image
import numpy as np
from einops import rearrange

from akira.utils.bokeh_utils import BokehRenderer
import math
from torch.distributions import Exponential

# reference ImageMagick inv barrel distortion
# @deprecated
# def apply_distortion_with_wand(np_img, A, B, C, D=1.0):
#     # convert to wand image
#     img = Image.from_array(np_img)
#     img.virtual_pixel = "transparent"
#     img.distort("barrel_inverse", (A, B, C, D))
#     # img.save(filename='checks_barrel.png')
#     # convert to opencv/numpy array format
#     img_opencv = np.array(img)
#     return img_opencv

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


# input size is [B, C, H, W]
def center_crop(image, rate, resize=True):
    n_image = image.shape[0]
    h, w = image.shape[-2:]
    crop_h, crop_w = int(h * rate), int(w * rate)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    cropped = image[:, :, start_h : start_h + crop_h, start_w : start_w + crop_w]
    if resize:
        return torch.nn.functional.interpolate(
            cropped, size=(h, w), mode="bilinear", align_corners=False
        )
    return cropped
  
def gaussian_2d(x, y, x0, y0, sigma_x, sigma_y):
    """Compute 2D Gaussian value at (x, y) with mean (x0, y0) and standard deviations sigma_x, sigma_y."""
    return torch.exp(
        -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
    )


def generate_spline_trajectory(start, end, num_points, max_speed_rate, total_time):
    """
    Generates a smooth trajectory using cubic splines, iterating over waypoints and regenerating
    those that lead to excessive speeds until all conform to the maximum speed constraint.

    Args:
    start (float): Starting position of the trajectory.
    end (float): Ending position of the trajectory.
    num_points (int): Number of waypoints to generate, including start and end.
    max_speed_rate (float): Maximum allowable speed between any two waypoints rate.
    total_time (float): Total time from start to end, used to calculate speed limits.

    Returns:
    CubicSpline: A spline object representing the trajectory.
    """
    if end < start:
        start, end = end, start
    max_speed = max_speed_rate * (end - start) / total_time
    # Generate initial random times and positions
    times = np.linspace(0, total_time, num_points)
    positions = np.random.uniform(
        low=min(start, end), high=max(start, end), size=num_points
    )
    # positions[0], positions[-1] = start, end  # Fix the start and end points

    valid = False
    while not valid:
        valid = True
        speeds = [
            np.abs(positions[i] - positions[i - 1]) / (times[i] - times[i - 1])
            for i in range(1, len(times))
        ]
        re_gen_mask = np.array(speeds) > max_speed
        if re_gen_mask.any():
            valid = False
            for i in np.where(re_gen_mask)[0]:
                positions[i] = np.random.uniform(
                    low=min(start, end), high=max(start, end)
                )
    cs = CubicSpline(times, positions)

    return cs, times, positions


# optical camera information
# intrinsic and distortion information
# a helper for camera rays generation
class OpticCameraInfo:
    """
    Class representing the optical camera information.

    Args:
      fx (float): The focal length in the x-direction.
      fy (float): The focal length in the y-direction.
      cx (float): The x-coordinate of the principal point.
      cy (float): The y-coordinate of the principal point.
      w (int): The width of the camera image.
      h (int): The height of the camera image.
      A (float): The first distortion parameter.
      B (float): The second distortion parameter.
      C (float): The third distortion parameter.
      D (float, optional): The fourth distortion parameter. Defaults to 1.0.
    """

    def __init__(self, fx, fy, cx, cy, w, h, ori_w, ori_h, A=0.0, B=0.0, C=0.0, D=1.0):
        self.w = w
        self.h = h

        self.ori_w = ori_w
        self.ori_h = ori_h

        # intrinsic
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # distortion
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # bokeh related
        self.bokeh_K = 0.0  # default K is zero is no bokeh
        self.bokeh_focus_x = 0.0
        self.bokeh_focus_y = 0.0
        self.bokeh_disp_focus = 0.0  # is this necessary?
        self.bokeh_defocus_scale = 10.0  # almost constant

        self.crop_rate = 1.0

    def get_intrinsics(self):
        """
        Get the intrinsic camera parameters.

        Returns:
          torch.Tensor: A tensor containing the intrinsic parameters [fx, fy, cx, cy].
        """
        return (self.fx, self.fy, self.cx, self.cy)

    def get_distortion_params(self, type="inv_barrel"):
        """
        Get the distortion parameters.

        Args:
          type (str, optional): The type of distortion. Defaults to "inv_barrel".

        Returns:
          torch.Tensor: A tensor containing the distortion parameters [A, B, C, D].
        """
        if type != "inv_barrel":
            raise NotImplementedError(f"{type} is not implemented")
        return (self.A, self.B, self.C, self.D)

    def get_distortion_mesh_grid(
        self, W, H, normalize=True, device="cpu", compute_max_crop_rate=False
    ):
        """
        Get the distortion mesh grid.

        Args:
          normalize (bool, optional): Whether to normalize the mesh grid to [-1,1]. Defaults to True.
          device (str, optional): Defaults to "cpu".

        Returns:
          torch.Tensor: A tensor representing the distortion mesh grid.
        """
        aspect_ratio = W / H
        x = torch.linspace(-1 * aspect_ratio, 1 * aspect_ratio, steps=W, device=device)
        y = torch.linspace(-1, 1, steps=H, device=device)
        y, x = torch.meshgrid(y, x)
        r = torch.sqrt(x**2 + y**2)
        x_distorted = x / (self.A * r**3 + self.B * r**2 + self.C * r + self.D)
        y_distorted = y / (self.A * r**3 + self.B * r**2 + self.C * r + self.D)
        if normalize:
            x_distorted = x_distorted / aspect_ratio
        grid = torch.stack([x_distorted, y_distorted], dim=-1).to(device)

        if compute_max_crop_rate:
            grid = grid.unsqueeze(0).repeat(1, 1, 1, 1)
            alpha_layer = torch.ones((1, 1, H, W), device=device)
            mask_crop = torch.nn.functional.grid_sample(
                alpha_layer,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            _mask_crop = mask_crop.clone()
            _crop_rate = 1.0
            while _mask_crop.all() != 1:
                _crop_rate = _crop_rate - 0.05
                _mask_crop = center_crop(mask_crop, _crop_rate, resize=False)
            return grid, _crop_rate
        return grid, 1.0


# one camera pose, for each it has a optic info
class Camera:
    def __init__(self, entry=None, args={}):
        self.oc_info = None
        self.w2c_mat = np.eye(4)
        self.c2w_mat = np.eye(4)
        if entry is not None:
            self.set_from_entry(entry, args)

    # Question: here or in the opti_camera_info?
    def rectify_intrinsic(self, samp_width, samp_height, ori_w, ori_h):
        sample_wh_ratio = samp_width / samp_height
        ori_wh_ratio = ori_w / ori_h
        # rescale fx
        if ori_wh_ratio > sample_wh_ratio:
            self.oc_info.fx = (
                ori_wh_ratio * self.oc_info.fx / sample_wh_ratio
            ) * samp_width
            self.oc_info.fy = self.oc_info.fy * samp_height
        else:
            self.oc_info.fy = (
                sample_wh_ratio * self.oc_info.fy / ori_wh_ratio
            ) * samp_height
            self.oc_info.fx = self.oc_info.fx * samp_width

        self.oc_info.cx = self.oc_info.cx * samp_width
        self.oc_info.cy = self.oc_info.cy * samp_height

    def set_from_entry(
        self,
        entry,
        args,
    ):
        width, height = args.image_width, args.image_height
        ori_width, ori_height = args.ori_width, args.ori_height

        fx, fy, cx, cy = entry[1:5]
        self.oc_info = OpticCameraInfo(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            w=width,
            h=height,
            ori_w=ori_width,
            ori_h=ori_height,
        )

        if args.is_rectify_intrinsic:
            self.rectify_intrinsic(width, height, ori_width, ori_height)

        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)

    # TODO: ugly! find a better way
    def get_w2c_torch(self):
        return torch.tensor(np.linalg.inv(self.c2w_mat), dtype=torch.float32)

    def get_c2w_torch(self):
        return torch.tensor(self.c2w_mat, dtype=torch.float32)
    
    def get_delta_c2w_torch(self):
        return torch.tensor(self.delta_c2w_mat, dtype=torch.float32)

    def update_c2w(self):
        self.c2w_mat = np.linalg.inv(self.w2c_mat)

    def set_w2c_mat(self, w2c_mat):
        self.w2c_mat = w2c_mat
        self.set_c2w_mat(np.linalg.inv(w2c_mat))
        self.delta_c2w_mat = self.get_delta_c2w_torch().numpy()

    def set_c2w_mat(self, c2w_mat):
        self.c2w_mat = c2w_mat
    
    def set_delta_c2w_mat(self, delta_c2w_mat):
        self.delta_c2w_mat = delta_c2w_mat
        
    def set_t(self, t):
        self.w2c_mat[:3, 3] = t
        self.update_c2w()
        
    def get_t(self):
        return self.w2c_mat[:3, 3]

    def set_R(self, R):
        self.w2c_mat[:3, :3] = R
        self.update_c2w()

    def output_as_entry(self, timestep=0):
        entry = [
            timestep,
            self.oc_info.fx,
            self.oc_info.fy,
            self.oc_info.cx,
            self.oc_info.cy,
            0,
            0,
        ]
        entry += list(self.w2c_mat[:3, :].reshape(-1))
        return entry

    def bokeh_condition(self, device, bokeh_reprez="sigmoid"):
        return Camera.compute_bokeh_condition(self.oc_info, device, bokeh_reprez=bokeh_reprez)

    @staticmethod
    def compute_bokeh_condition(oc_info, device, use_positional_embedding=False, bokeh_reprez="sigmoid"):
        fx, fy, cx, cy = oc_info.get_intrinsics()

        # distortion information
        distort_mesh, _ = oc_info.get_distortion_mesh_grid(
            W=oc_info.ori_w, H=oc_info.ori_h, normalize=True, device=device
        )

        oH = distort_mesh.shape[0]
        oW = distort_mesh.shape[1]

        # focus point information
        fp_pix_coord_y_pre_dist = int(((oc_info.bokeh_focus_y) + 1) / 2.0 * oH)
        fp_pix_coord_x_pre_dist = int(((oc_info.bokeh_focus_x) + 1) / 2.0 * oW)

        # focus point coordinate after distortion: -1, 1
        dist_fp_coordinate = distort_mesh[
            fp_pix_coord_y_pre_dist, fp_pix_coord_x_pre_dist
        ]

        distort_mesh = rearrange(distort_mesh, "h w c -> 1 c h w")
        # no resize as the focal length changes
        # so we just trim the outter pixels
        distort_mesh = center_crop(distort_mesh, oc_info.crop_rate, resize=False)
        distort_mesh = rearrange(distort_mesh, "1 c h w -> h w c")
        H = distort_mesh.shape[0]
        W = distort_mesh.shape[1]

        # div by crop rate directly as the orignal is -1, 1
        # dist_fp_coordinate = dist_fp_coordinate / oc_info.crop_rate
        dist_fp_coordinate[0] = dist_fp_coordinate[0] / (H / oH)
        dist_fp_coordinate[1] = dist_fp_coordinate[1] / (W / oW)

        # translate new_fp_coord to new_fp_pix_coord
        dist_fp_pix_coord_y = int((dist_fp_coordinate[1] + 1) / 2.0 * H)
        dist_fp_pix_coord_x = int((dist_fp_coordinate[0] + 1) / 2.0 * W)
        dist_crop_fp_pix_coord = [dist_fp_pix_coord_y, dist_fp_pix_coord_x]

        # relative vector to the focus point
        h_grid, w_grid = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=distort_mesh.dtype),
            torch.linspace(0, W - 1, W, device=device, dtype=distort_mesh.dtype),
        )
        dist_h = h_grid - dist_crop_fp_pix_coord[0]
        dist_w = w_grid - dist_crop_fp_pix_coord[1]
        aperture = oc_info.bokeh_K / oc_info.bokeh_defocus_scale

        # 3D map
        relative_map = torch.zeros((H, W, 3), device=device)
        relative_map[:, :, 0] = dist_h / H
        relative_map[:, :, 1] = dist_w / W

        # change the the multiplication of aperture to power operation
        if bokeh_reprez == "power":
            power_factor = (aperture / 10.0)
            relative_map[:, :, 2] = torch.pow(
            (torch.sqrt(dist_h**2 + dist_w**2) / math.sqrt(H**2 + W**2)),
            power_factor)
        elif bokeh_reprez == "inv_power":
            power_factor = min(1.0 / (aperture + 1e-5 / 10.0), 50)
            relative_map[:, :, 2] = torch.pow(
            (torch.sqrt(dist_h**2 + dist_w**2) / math.sqrt(H**2 + W**2)),
            power_factor)
        elif bokeh_reprez == "sigmoid":
            distance_map = torch.sqrt(dist_h**2 + dist_w**2) / math.sqrt(H**2 + W**2)
            distance_map = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min())        
            # sigmoid 0 is 0.5
            aperture_factor = 1.0/torch.sigmoid(torch.Tensor([aperture]))
            # standardize to -1, 1
            relative_map[:, :, 2] = 2*torch.pow(distance_map, aperture_factor) - 1.0
        elif bokeh_reprez == "sigmoida":
            distance_map = torch.sqrt(dist_h**2 + dist_w**2) / math.sqrt(H**2 + W**2)
            distance_map = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min())        
            aperture_factor = 1.0/(torch.sigmoid(torch.Tensor([aperture])))
            aperture_factor = torch.pow(aperture_factor-0.05, 2)
            # standardize to -1, 1
            relative_map[:, :, 2] = 2*torch.pow(distance_map, aperture_factor) - 1.0
        elif bokeh_reprez == "sigmoidb":
            distance_map = torch.sqrt(dist_h**2 + dist_w**2) / math.sqrt(H**2 + W**2)
            distance_map = (distance_map - distance_map.min()) / (distance_map.max() - distance_map.min())        
            aperture_factor = 1.0/((aperture+1.0) * torch.sigmoid(torch.Tensor([aperture])))
            # standardize to -1, 1
            relative_map[:, :, 2] = 2*torch.pow(distance_map, aperture_factor) - 1.0

        bokeh_embedding = rearrange(relative_map, "h w c-> c h w")

        # [for viz focus point]: rectangle map
        # gaussian_grid = torch.zeros((H, W), device=device)
        # gaussian_grid[dist_crop_fp_pix_coord[0]-10:dist_crop_fp_pix_coord[0]+10, dist_crop_fp_pix_coord[1]-10:dist_crop_fp_pix_coord[1]+10] = 1.0

        # gaussian_grid = gaussian_grid.repeat(3, 1, 1)

        if use_positional_embedding:
            print("error here, use_positional_embedding is not implemented")
            raise NotImplementedError("use_positional_embedding is not implemented")

        bokeh_embedding = rearrange(bokeh_embedding, "c h w -> 1 1 c h w")
        bokeh_embedding = torch.nn.functional.interpolate(
            bokeh_embedding.squeeze(0),
            size=(oc_info.h, oc_info.w),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(0)

        return bokeh_embedding

    def ray_condition(self, device, plucker_type="cctrl"):
        if "delta" in plucker_type:
            c2w = self.get_delta_c2w_torch().to(device)
        else:
            c2w = self.get_c2w_torch().to(device)
            
        c2w = c2w.unsqueeze(0).unsqueeze(0)  # [B(1), V(1), 4, 4]
        return Camera.compute_ray_condition(c2w, self.oc_info, device, plucker_type)

    @staticmethod
    def compute_ray_condition(c2w, oc_info, device, plucker_type="cctrl"):
        """we now compute this directly from distort_mesh
        if is_flip:
            j, i = custom_meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=dtype),
                torch.linspace(W - 1, 0, W, device=device, dtype=dtype),
            )
        else:
            j, i = custom_meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=dtype),
                torch.linspace(0, W - 1, W, device=device, dtype=dtype),
            )
        i = i.reshape([1, 1, H * W]) + 0.5 # [B(1), V(1), HxW] # why 0.5 here?
        j = j.reshape([1, 1, H * W]) + 0.5  # [B(1), V(1), HxW]
        zs = torch.ones_like(i)  # [B(1), V(1), HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys) # why expand here?
        """
        c2w = c2w.to(device)  # [B(1), V(1), 4, 4]

        fx, fy, cx, cy = oc_info.get_intrinsics()

        distort_mesh, _ = oc_info.get_distortion_mesh_grid(
            W=oc_info.ori_w, H=oc_info.ori_h, normalize=True, device=device
        )

        distort_mesh = rearrange(distort_mesh, "h w c -> 1 c h w")
        distort_mesh = center_crop(distort_mesh, oc_info.crop_rate, resize=False)
        distort_mesh = rearrange(distort_mesh, "1 c h w -> h w c")
        H = distort_mesh.shape[0]
        W = distort_mesh.shape[1]

        zs = torch.ones_like(distort_mesh[:, :, 0]).reshape(
            1, 1, H * W
        )  # [B(1), V(1), HxW]
        xs = ((distort_mesh[:, :, 0] * cx) / fx).reshape(1, 1, H * W)
        ys = ((distort_mesh[:, :, 1] * cy) / fy).reshape(1, 1, H * W)

        directions = torch.stack((xs, ys, zs), dim=-1)  # B(1), V(1), HW, 3
        directions = directions / directions.norm(
            dim=-1, keepdim=True
        )  # B(1), V(1), HW, 3

        # R^-T K^-1 u : writen in transposed form
        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B(1), V(1), HW, 3
        rays_o = c2w[..., :3, 3]  # B(1), V(1), 3
        rays_o = rays_o[:, :, None].expand_as(rays_d)  # B(1), V(1), HW, 3

        # o x direct
        rays_dxo = torch.cross(rays_o, rays_d)  # B(1), V(1), HW, 3
        # rays_dxno = torch.cross(rays_o_nr, rays_d)  # B(1), V(1), HW, 3

        # default cctrl from cameractrl
        if "cctrl" in plucker_type :
            plucker = torch.cat([rays_dxo, rays_d], dim=-1)
        elif "decoupled" in plucker_type:
            plucker = torch.cat([rays_dxo, directions], dim=-1)

        plucker = plucker.reshape(c2w.shape[1], H, W, 6).unsqueeze(0)
        # B(1), V(1), H, W, 6

        plucker = rearrange(plucker, "b v h w c -> b v c h w")
        # B is one anyway
        plucker = torch.nn.functional.interpolate(
            plucker.squeeze(0),
            size=(oc_info.h, oc_info.w),
            mode="bilinear",
            align_corners=False,
        ).unsqueeze(0)

        return plucker

    @staticmethod
    def get_relative_pose(cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        cam_to_origin = 0
        target_cam_c2w = np.array(
            [[1, 0, 0, 0], [0, 1, 0, -cam_to_origin], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [
            target_cam_c2w,
        ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses
    
    @staticmethod
    def get_delta_pose(cam_params):
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        delta_transforms = [np.eye(4)] + [np.linalg.inv(abs_c2ws[i - 1]) @ abs_c2ws[i] for i in range(1, len(abs_c2ws))]
        delta_transforms = np.array(delta_transforms, dtype=np.float32)
        return delta_transforms

class Trajectory:
    def __init__(self, args=None):
        self.args = args

    def read_from_file(
        self,
        file_path,
        args=None,
        skip_interval=1,
        read_lines=16,
    ):
        if args:
            self.args = args

        with open(file_path, "r") as f:
            lines = f.readlines()

        # recitfied
        self.cameras = [
            Camera([float(x) for x in pose.strip().split(" ")], args)
            for pose in lines[1:(read_lines*skip_interval)+1:skip_interval]
        ]
        if not (len(self.cameras) == self.args.video_length):
          print(f"Expected {self.args.video_length} cameras, got {len(self.cameras)}")
          return False
        return True

    def read_from_camera_list(self, camera_list, args=None):
        if args:
            self.args = args
        self.cameras = [Camera([float(x) for x in pose], args) for pose in camera_list]

    def write_to_file(self, file_path):
        with open(file_path, "w") as f:
            f.write(f"{len(self.cameras)}\n")
            for i, cam in enumerate(self.cameras):
                f.write(" ".join([str(x) for x in cam.output_as_entry(i)]) + "\n")
                
    def apply_relative_pose(self):
        c2w_poses = Camera.get_relative_pose(self.cameras)
        delta_c2w_poses = Camera.get_delta_pose(self.cameras)
        
        for i, cam in enumerate(self.cameras):
            cam.set_c2w_mat(c2w_poses[i])
            cam.set_delta_c2w_mat(delta_c2w_poses[i])
    
    def clear_all_pose(self):
        for cam in self.cameras:
            cam.set_w2c_mat(np.eye(4))
            cam.set_delta_c2w_mat(np.eye(4))    
    
    def rand_focal_length(self, rate_low, rate_high, speed_rate=3, simple_mode=False, static_mode=False):
        num_points, max_speed_rate, total_time = (
            len(self.cameras),
            speed_rate,
            len(self.cameras),
        )
        if simple_mode:
          rand_a, rand_b = 0, 0
          while abs(rand_a - rand_b) < 1.0:
            rand_a = np.random.uniform(rate_low, rate_high)
            rand_b = np.random.uniform(rate_low, rate_high)
          
          # in static mode, the focal length is the same for all cameras
          if static_mode: 
              rates = rand_a * np.ones(num_points)
          else:
              rates = np.linspace(rand_a, rand_b, num_points)
        else:
            _, _, rates = generate_spline_trajectory(
            rate_low, rate_high, num_points, max_speed_rate, total_time
          )

        for i, cam in enumerate(self.cameras):
            cam.oc_info.fx = cam.oc_info.fx * rates[i]
            cam.oc_info.fy = cam.oc_info.fy * rates[i]
            cam.oc_info.crop_rate = cam.oc_info.crop_rate / rates[i]
            
    def rand_intrinsics_distortions_simple(self, bound):
        A = np.random.uniform(-bound, bound)
        As = A * np.ones(len(self.cameras))
        B = np.random.uniform(-bound, bound)
        Bs = B * np.ones(len(self.cameras))
        C =  np.random.uniform(-bound, bound)
        Cs = C * np.ones(len(self.cameras))

        # fix the focal length according to the distortion
        # apply distortion
        for i, cam in enumerate(self.cameras):
            cam.oc_info.A = As[i]
            cam.oc_info.B = Bs[i]
            cam.oc_info.C = Cs[i]

        for i, cam in enumerate(self.cameras):
            H, W = cam.oc_info.ori_h, cam.oc_info.ori_w
            distort_mesh, max_crop_rate = cam.oc_info.get_distortion_mesh_grid(
                W=W, H=H, normalize=True, device="cpu", compute_max_crop_rate=True
            )
            max_crop_rate-=0.02
            cam.oc_info.crop_rate = max_crop_rate
            cam.oc_info.fx = cam.oc_info.fx / cam.oc_info.crop_rate
            cam.oc_info.fy = cam.oc_info.fy / cam.oc_info.crop_rate
            
    def rand_intrinsics_distortions(self, h_bound, l_bound, speed_rate=0.05):
        num_points, max_speed_rate, total_time = (
            len(self.cameras),
            speed_rate,
            len(self.cameras),
        )
        _, _, As = generate_spline_trajectory(
            h_bound, l_bound, num_points, max_speed_rate, total_time
        )
        _, _, Bs = generate_spline_trajectory(
            h_bound, l_bound, num_points, max_speed_rate, total_time
        )
        _, _, Cs = generate_spline_trajectory(
            h_bound, l_bound, num_points, max_speed_rate, total_time
        )

        # D is often 1.0

        # fix the focal length according to the distortion
        # apply distortion
        for i, cam in enumerate(self.cameras):
            cam.oc_info.A = As[i]
            cam.oc_info.B = Bs[i]
            cam.oc_info.C = Cs[i]

        for i, cam in enumerate(self.cameras):
            H, W = cam.oc_info.ori_h, cam.oc_info.ori_w
            distort_mesh, max_crop_rate = cam.oc_info.get_distortion_mesh_grid(
                W=W, H=H, normalize=True, device="cpu", compute_max_crop_rate=True
            )
            cam.oc_info.crop_rate = max_crop_rate
            cam.oc_info.fx = cam.oc_info.fx / cam.oc_info.crop_rate
            cam.oc_info.fy = cam.oc_info.fy / cam.oc_info.crop_rate

    def distort_imgs(self, imgs, device):
        distorted_images = []
        for idx, img in enumerate(imgs):
            img = img.unsqueeze(0)
            distorted_image, tranparent_layer, grid = inv_barrel_distortion(
                img,
                self.cameras[idx].oc_info,
                device,
                seperate_alpha=True,
                crop_rate=self.cameras[idx].oc_info.crop_rate,
                crop_distortion=True,
            )
            distorted_images.append(distorted_image)

        distorted_images = torch.vstack(distorted_images)
        return distorted_images

    ##############################
    ### Bokeh related functions ###
    ##############################

    def bokeh_imgs(self, imgs, disp, device):

        bokeh_imgs = []
        defocus_maps = []
        for i, img in enumerate(imgs):
            bokeh_args = {
                "K": self.cameras[i].oc_info.bokeh_K,
                "disp_focus_point": [
                    self.cameras[i].oc_info.bokeh_focus_y,
                    self.cameras[i].oc_info.bokeh_focus_x,
                ],
            }
            bokeh_renderer = BokehRenderer(bokeh_args, device="cuda")
            bokeh_img, defocus_map = bokeh_renderer.render(img, disp[i])
            bokeh_imgs.append(bokeh_img)
            defocus_maps.append(defocus_map)
        bokeh_imgs = torch.stack(bokeh_imgs)
        defocus_maps = torch.stack(defocus_maps)
        return bokeh_imgs, defocus_maps

    # random bokeh images
    def rand_bokeh(self, fp_bound, fp_speed_rate=0.1, K_bound=100, K_speed_rate=1):
        # args need to randomize,
        # focus point of x, y
        # K in terms of bokeh level (aperature), assmuing max K is 100

        num_points, total_time = len(self.cameras), len(self.cameras)

        _, _, Xs = generate_spline_trajectory(
            -fp_bound, fp_bound, num_points, fp_speed_rate, total_time
        )  # in total -1 to 1

        _, _, Ys = generate_spline_trajectory(
            -fp_bound, fp_bound, num_points, fp_speed_rate, total_time
        )
        
        rand_K_bound = np.random.uniform(0, K_bound)

        _, _, bKs = generate_spline_trajectory(
            0, rand_K_bound, num_points, K_speed_rate, total_time
        )

        for i, cam in enumerate(self.cameras):
            cam.oc_info.bokeh_focus_x = Xs[i].item()
            cam.oc_info.bokeh_focus_y = Ys[i].item()
            cam.oc_info.bokeh_K = bKs[i] * (
                self.cameras[i].oc_info.h / self.cameras[i].oc_info.ori_h
            )  # normalise the K wrt to original_h

    # random bokeh images
    def rand_bokeh_simple(self, fp_bound, fp_speed_rate=0.1, K_bound=100, K_speed_rate=1):
        delta = 100/len(self.cameras)
        bokeh = 0
        for i, cam in enumerate(self.cameras):
            cam.oc_info.bokeh_focus_x = 0.0 
            cam.oc_info.bokeh_focus_y = 0.0
            cam.oc_info.bokeh_K = bokeh * (
                self.cameras[i].oc_info.h / self.cameras[i].oc_info.ori_h
            )  # normalise the K wrt to original_h
            bokeh += delta
            
            
    ##############################

    def get_plucker_embeddings(self, device):
        plucker_embeddings = [
            cam.ray_condition(device, plucker_type=self.args.plucker_type)
            for cam in self.cameras
        ]
        plucker_embeddings = (
            torch.cat(plucker_embeddings, dim=1).contiguous().to(device)
        )

        return plucker_embeddings  # B(1), V(16), 6, H, W

    def get_plucker_embeddings_bokehs(self, device, use_bokeh=True, bokeh_reprez="sigmoid"):
        plucker_embeddings = self.get_plucker_embeddings(device)  # B(1), V(16), 6, H, W
        
        if not use_bokeh:
            return plucker_embeddings
          
        bokeh_plucker_embeddings = [cam.bokeh_condition(device, bokeh_reprez) for cam in self.cameras]
        bokeh_plucker_embeddings = (
            torch.cat(bokeh_plucker_embeddings, dim=1).contiguous().to(device)
        )  # B(1), V(16), 1, H, W

        # add bokeh information to the plucker embeddings
        plucker_embeddings = torch.cat(
            [plucker_embeddings, bokeh_plucker_embeddings], dim=2
        )

        return plucker_embeddings

    # used for optimsation
    def compute_ray_condition_from_multiple_cameras(
        self, multiCamTrans, device, plucker_type="cctrl"
    ):
        assert len(multiCamTrans) == len(self.cameras)
        eye_pose = torch.eye(4)
        c2ws = [multiCamTrans(idx, eye_pose) for idx in range(len(multiCamTrans))]
        rays = [
            Camera.compute_ray_condition(
                c2w.unsqueeze(0).unsqueeze(0),
                self.cameras[idx].oc_info,
                device,
                plucker_type,
            )
            for idx, c2w in enumerate(c2ws)
        ]
        rays = torch.cat(rays, dim=1).contiguous().to(device)
        return rays


# torch version of the ImageMagick inv barrel distortion
# W and H are from image_tensor
def inv_barrel_distortion(
    img_tensor,
    OpticCameraInfo,
    device="cpu",
    seperate_alpha=False,
    crop_rate=-1.0,
    crop_distortion=False,
):
    assert img_tensor.dim() == 4, f"Expected 4D input tensor, got {img_tensor.dim()}"
    # Get dimensions
    BS, _, height, width = img_tensor.shape
    img_tensor = img_tensor.to(device)

    # Get the distortion mesh grid
    grid, max_crop = OpticCameraInfo.get_distortion_mesh_grid(
        W=width, H=height, normalize=True, device=device
    )
    grid = grid.unsqueeze(0).repeat(BS, 1, 1, 1)

    alpha_layer = torch.ones((BS, 1, height, width), device=device)
    # Note: for grid_sample the grid is [-1, 1]
    distorted_image = torch.nn.functional.grid_sample(
        img_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    # add mask for easier usage
    tranparent_layer = torch.nn.functional.grid_sample(
        alpha_layer, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    if seperate_alpha == False:
        distorted_image = torch.cat([distorted_image, tranparent_layer], dim=1)

    if crop_distortion:
        final_crop_rate = crop_rate if crop_rate > 0.0 else max_crop
        distorted_image = center_crop(distorted_image, final_crop_rate)

    return distorted_image, tranparent_layer, grid


"""
# TrajectoryEditor class
# Use this class to edit the trajectory of the camera
# the trajectory class should be already defined
"""
class TrajectoryEditor:
    def _apply_mode(self, traj, mode):
        if mode not in ["override", "blend"]:
            raise ValueError("mode should be override or blend")
        if mode == "override":
            traj.clear_all_pose()
            
    def pure_translation(self, traj, axis='z', distance=0, speed_rate=0.2, mode="override"):
        """
        Apply pure translation along specified axis.
        Args:
            traj: trajectory object
            axis: 'x', 'y', or 'z' for translation direction
            distance: initial distance
            speed_rate: rate of distance change per frame
            mode: "override" or "blend"
        """
        self._apply_mode(traj, mode)
        init_distance = distance
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        
        for i, cam in enumerate(traj.cameras):
            previous_distance = cam.get_t()[axis_idx]
            new_t = np.zeros(3)
            new_t[axis_idx] = init_distance + previous_distance
            cam.set_t(new_t)
            cam.update_c2w()
            init_distance += speed_rate
            
    def pure_rotation(self, traj, axis='y', angle=0, speed_rate=-0.06, mode="override"):
        """
        Apply pure rotation around specified axis.
        Args:
            traj: trajectory object
            axis: 'x', 'y', or 'z' for rotation axis
            angle: initial angle
            speed_rate: rate of angle change per frame
            mode: "override" or "blend"
        """
        self._apply_mode(traj, mode)
        
        # Define rotation matrices for each axis
        def get_rotation_matrix(angle, axis):
            if axis == 'x':
                return np.array([
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]
                ])
            elif axis == 'y':
                return np.array([
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]
                ])
            else:  # z-axis
                return np.array([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]
                ])
        
        for cam in traj.cameras:
            angle = angle + speed_rate
            cam.set_R(get_rotation_matrix(angle, axis.lower()))