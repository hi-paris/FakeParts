import torch
import torch.nn as nn


def vec2ss_matrix(x: torch.Tensor):
    """Vector to skew-symetric matrix."""
    ss_matrix = torch.zeros((3, 3), device=x.device)
    ss_matrix[0, 1] = -x[2]
    ss_matrix[0, 2] = x[1]
    ss_matrix[1, 0] = x[2]
    ss_matrix[1, 2] = -x[0]
    ss_matrix[2, 0] = -x[1]
    ss_matrix[2, 1] = x[0]

    return ss_matrix

class MultiCameraTransform(nn.Module):
    """Multiple camera transformation module using Rodrigues formula."""
    def __init__(
        self, num_cameras: int = 16, refocalize: bool = True, device="cuda"
    ):
        super(MultiCameraTransform, self).__init__()

        self.num_cameras = num_cameras

        # Rotvec
        w_s = [
            nn.Parameter(torch.normal(0.0, 1e-8, size=(3,))) for _ in range(num_cameras)
        ]
        self.w = nn.ParameterList(w_s)
        # Velocity t
        v_s = [
            nn.Parameter(torch.normal(0.0, 1e-8, size=(3,))) for _ in range(num_cameras)
        ]
        self.v = nn.ParameterList(v_s)
        # Angles
        theta_s = [
            nn.Parameter(torch.normal(0.0, 1e-8, size=(1,))) for _ in range(num_cameras)
        ]
        self.theta = nn.ParameterList(theta_s)
        # Gather all spatial parameters
        self.spatial = nn.ParameterList([self.v, self.w, self.theta])
        
        # Focal is defined as scaling factor [0,1]
        if refocalize:
            self.focal_factor_param = nn.ParameterList(
                [nn.Parameter(-torch.ones(1)) for _ in range(num_cameras)]
            )
        if not refocalize:
            self.focal_factor_param = -torch.ones((num_cameras, 1, 1))

        self.interpolation_ratios = torch.linspace(0, 1, num_cameras).to(device="cuda")
    
    def __len__(self):
        return self.num_cameras
      
    def forward(self, camera_index: int, x: torch.Tensor) -> torch.Tensor:
        w_skewsym = vec2ss_matrix(self.w[camera_index])

        exp_i = torch.zeros((4, 4), device=x.device)
        exp_i[:3, :3] = (
            torch.eye(3, device=x.device)
            + torch.sin(self.theta[camera_index]) * w_skewsym
            + (1 - torch.cos(self.theta[camera_index]))
            * torch.matmul(w_skewsym, w_skewsym)
        )
        exp_i[:3, 3] = torch.matmul(
            torch.eye(3, device=x.device) * self.theta[camera_index]
            + (1 - torch.cos(self.theta[camera_index])) * w_skewsym
            + (self.theta[camera_index] - torch.sin(self.theta[camera_index]))
            * torch.matmul(w_skewsym, w_skewsym),
            self.v[camera_index],
        )
        exp_i[3, 3] = 1.0
        T_i = torch.matmul(exp_i, x)

        return T_i

    def get_pose(self, camera_index: int) -> torch.Tensor:
        w_cam = self.spatial[0][camera_index]
        v_cam = self.spatial[1][camera_index]
        theta_cam = self.spatial[2][camera_index]
        return w_cam, v_cam, theta_cam

    def set_focal_factor(self, focal_factor):
        for param in self.focal_factor_param:
            param.data[:] = focal_factor

    def get_focal_factor(self, camera_index: int):
        return self.focal_factor_param[camera_index]

    def set_intrinsic(self, intrinsic, H, W):
        self.H = H
        self.W = W
        self.intrinsic = torch.tensor(intrinsic)
        self.intrinsic[2] = self.W / 2
        self.intrinsic[3] = self.H / 2
        self.intrinsic.requires_grad = False

    def get_intrinsic(self, camera_index: int):
        if self.focal_factor_param[0].device != self.intrinsic.device:
            self.intrinsic = self.intrinsic.to(device=self.focal_factor_param[0].device)
        current_intrinsic = self.intrinsic.clone().detach()
        current_intrinsic[0] = self.intrinsic[0] * self.get_focal_factor(
            camera_index
        ).squeeze(0)
        current_intrinsic[1] = self.intrinsic[1] * self.get_focal_factor(
            camera_index
        ).squeeze(0)
        
        # intrinsic as fx fy cx cy
        return current_intrinsic

    def get_intrinsic_from_focal_factor(self, focal_factor):
        current_intrinsic = self.intrinsic.clone().detach()
        current_intrinsic[0] = self.intrinsic[0] * focal_factor
        current_intrinsic[1] = self.intrinsic[1] * focal_factor
        return current_intrinsic

    # could be useless
    def clip_parameters(self, spatial_clip_size):
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "spatial" in name:
                    torch.nn.utils.clip_grad_norm_(self.spatial, spatial_clip_size)
                if "focal" in name:
                    for param in self.focal_factor_param:
                        torch.clamp_(param.data, min=0.1, max=6.0)

    # could be useless
    def clip_parameters_schedule(
        self, spatial_clip_size, camera_idx_list, factor
    ):
        for camera_idx in range(self.num_cameras):
            if camera_idx in camera_idx_list:
                for spatial_idx in range(3):
                    torch.nn.utils.clip_grad_norm_(
                        self.spatial[spatial_idx][camera_idx],
                        factor * spatial_clip_size,
                    )
            else:
                for spatial_idx in range(3):
                    torch.nn.utils.clip_grad_norm_(
                        self.spatial[spatial_idx][camera_idx],
                        spatial_clip_size,
                    )
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "focal" in name:
                    for subparam in self.focal_factor_param:
                        torch.clamp_(subparam.data, min=0.1, max=6.0)

    def get_ratios(self) -> torch.Tensor:
        return self.interpolation_ratios