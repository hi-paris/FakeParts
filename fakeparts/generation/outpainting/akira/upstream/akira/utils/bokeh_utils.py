import numpy as np
import torch
import torch.nn.functional as F
from akira.utils.bokeh_render_cupy_utils import ModuleRenderScatter
# from akira.utils.bokeh_render_torch_native_aprox_utils import ModuleRenderScatter

class BokehRenderer:
    def __init__(self, args, device=None):
      self.args = args
      self.device = device
      self.renderer = ModuleRenderScatter().to(self.device)
      
      # mandatory arguments
      self.K = args["K"]
      self.disp_focus_point = args["disp_focus_point"]
      
      # gamma optional
      self.gamma = args.get("gamma", 4)
      self.defocus_scale = args.get("defocus_scale", 10.0)
      
      # highlight optional, default True
      self.highlight = args.get("highlight", True)
      self.highlight_RGB_threshold = args.get("highlight_RGB_threshold", 220.0 / 255.0)
      self.highlight_enhance_ratio = args.get("highlight_enhance_ratio", 0.4)

    def render(self, image, disp):
      # typecast
      if image.dtype == np.uint8:
        image = np.float32(image) / 255.0
      elif image.dtype == np.float32:
        image = image.astype(np.float32)
        
      if image.shape[0] == 3:
        image = image.permute(1, 2, 0)
      
      # change tensor to numpy
      if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
      if isinstance(disp, torch.Tensor):
        disp = disp.cpu().numpy()
         

      disp = (disp) / 255.0
      disp = np.float32(disp)
      # resize disp to image size
      disp = F.interpolate(torch.from_numpy(disp).unsqueeze(0).unsqueeze(0), size=(image.shape[0], image.shape[1]), mode='bilinear', align_corners=False).squeeze().numpy()     
      disp = np.float32(disp)

      disp_fp_weight_y = (self.disp_focus_point[0] + 1) / 2
      disp_fp_weight_x = (self.disp_focus_point[1] + 1) / 2

      focus_y = int(disp_fp_weight_y * image.shape[0])
      focus_x = int(disp_fp_weight_x * image.shape[1])
      
      # take a aera of focus
      y_low = max(0, focus_y - 5)
      y_high = min(image.shape[0], focus_y + 5)
      x_low = max(0, focus_x - 5)
      x_high = min(image.shape[1], focus_x + 5)
      disp_focus = disp[y_low:y_high, x_low:x_high].mean()
      
      # single point focus
      # disp_focus = disp[focus_y, focus_x]
      focus_map = np.zeros_like(disp)
      focus_map[y_low:y_high, x_low:x_high] = 1.0
      focus_map = torch.from_numpy(focus_map).unsqueeze(0).cpu()
      
      # Handle highlights if enabled
      if self.highlight:
        image = self.apply_highlights(image, disp, disp_focus)

      defocus = self.K * (disp - disp_focus) / self.defocus_scale
      image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
      defocus = torch.from_numpy(defocus).unsqueeze(0).unsqueeze(0).to(self.device).to(torch.float32)
      with torch.no_grad():
        bokeh_classical = self.pipeline(image, defocus, self.gamma)

      return bokeh_classical[0].cpu(), focus_map

    def pipeline(self, image, defocus, gamma):
        bokeh_classical, defocus_dilate = self.renderer(image**gamma, defocus*self.defocus_scale)
        bokeh_classical = bokeh_classical ** (1/gamma)
        return bokeh_classical.clamp(0, 1)

    def apply_highlights(self, image, disp, focus):
        mask1 = np.clip(np.tanh(200 * (np.abs(disp - focus)**2 - 0.01)), 0, 1)[..., np.newaxis]
        mask2 = np.clip(np.tanh(10 * (image - self.highlight_RGB_threshold)), 0, 1)
        mask = mask1 * mask2
        return image * (1 + mask * self.highlight_enhance_ratio)
