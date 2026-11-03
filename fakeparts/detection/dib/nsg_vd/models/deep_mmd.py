import torch
import torch.nn as nn
from utils.mmd_utils import MMDu
from models.discriminators import I3DDiscriminator

class deep_MMD(nn.Module):
    def __init__(self, discriminator, sigma=1000, sigma0=0.1, epsilon=10, img_size=224, is_yy_zero=False, is_smooth=True):
        super(deep_MMD, self).__init__()
        self.img_size = img_size
        self.is_yy_zero = is_yy_zero
        self.is_smooth = is_smooth
        self.epsilonOPT = nn.Parameter(torch.log(torch.rand(1) * 10 ** (-epsilon)), requires_grad=True)
        self.sigmaOPT = nn.Parameter(torch.sqrt(torch.tensor(2 * self.img_size * self.img_size * sigma)), requires_grad=True)
        self.sigma0OPT = nn.Parameter(torch.sqrt(torch.tensor(sigma0)), requires_grad=True)
        
        self.register_buffer('mmd_value', torch.tensor(0.0)) 
        self.register_buffer('ep', torch.tensor(0.0))
        self.register_buffer('sigma', torch.tensor(0.0))
        self.register_buffer('sigma0_u', torch.tensor(0.0))
        self.register_buffer('STAT_u', torch.tensor(0.0))
        
        self.net = discriminator

    def forward(self, x, input_shape, writer=None, global_step=None):
        _, outputs = self.net(x, out_feature=True)
        
        ep = torch.exp(self.epsilonOPT) / (1 + torch.exp(self.epsilonOPT))
        sigma = self.sigmaOPT ** 2
        sigma0_u = self.sigma0OPT ** 2
        TEMP = MMDu(outputs, input_shape, x.view(x.shape[0],-1), sigma, sigma0_u, ep, is_yy_zero=self.is_yy_zero, is_smooth=self.is_smooth, writer=writer, global_step=global_step)
        
        return TEMP, ep.item(), sigma.item(), sigma0_u.item()

    def set_info(self, mmd_value, ep, sigma, sigma0_u, STAT_u):
        self.mmd_value.copy_(torch.tensor(mmd_value))
        self.ep.copy_(torch.tensor(ep))
        self.sigma.copy_(torch.tensor(sigma))
        self.sigma0_u.copy_(torch.tensor(sigma0_u))
        self.STAT_u.copy_(torch.tensor(STAT_u))
        
if __name__ == "__main__":
    from torchinfo import summary
    discriminator = I3DDiscriminator(9, 8)
    model = deep_MMD(discriminator=discriminator, sigma=1, sigma0=1, epsilon=1)
    summary(model)