import torch
import torch.nn as nn
from guided_diffusion_modules.unet import UNet
from config import shape


class Diffusion_model(nn.Module):
    def __init__(self, device, beta_1, beta_T, T):
        '''
        The epsilon predictor of diffusion process.

        beta_1    : beta_1 of diffusion process
        beta_T    : beta_T of diffusion process
        T         : Diffusion Steps
        input_dim : a dimension of data

        '''

        super().__init__()
        self.device = device
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        self.backbone = UNet(
                image_size=shape[1],
                in_channel=2 * shape[0],
                inner_channel=64,
                out_channel=3,
                res_blocks=3,
                attn_res=[8],
            ).to(device=device)

        self.to(device=self.device)

    def loss_fn(self, x, img=None, idx=None):
        output, epsilon, alpha_bar = self.forward(x, img, idx=idx, get_target=True)
        loss = (output - epsilon).square().mean()
        return loss

    def forward(self, x, img=None, idx=None, get_target=False):
        if idx == None:
            idx = torch.randint(0, len(self.alpha_bars), (x.size(0),)).to(device=self.device)
            used_alpha_bars = self.alpha_bars[idx][:, None, None, None]
            epsilon = torch.randn_like(x)
            x_tilde = torch.sqrt(used_alpha_bars) * x + torch.sqrt(1 - used_alpha_bars) * epsilon

        else:
            idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device=self.device).long()
            x_tilde = x

        if img is None:
            raise ValueError('If you want to use conditional model, you must specify img')

        x_tilde = torch.cat([img, x_tilde], dim=1)

        output = self.backbone(x_tilde, idx)

        return (output, epsilon, used_alpha_bars) if get_target else output