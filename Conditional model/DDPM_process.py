import torch
from Diffusion_model import Diffusion_model
from SSIM import SSIM
from torch.nn import functional as F


class DiffusionProcessDDPM():
    def __init__(self, beta_1, beta_T, T, diffusion_fn: Diffusion_model, device, shape):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        '''
        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])
        self.shape = shape

        self.diffusion_fn = diffusion_fn
        self.device = device

    def _conditional_one_diffusion_step(self, img):
        total_steps = len(self.alpha_bars)
        for i in range(3):
            diffusion_process = list(range(0, total_steps, 1)) + [total_steps - 1]
            y = torch.sqrt(self.alpha_bars[total_steps - 1]) * img + torch.randn_like(img) * torch.sqrt(
                1 - self.alpha_bars[total_steps - 1])
            for idx in reversed(diffusion_process):
                noise = torch.zeros_like(y) if idx == 0 else torch.randn_like(y)
                predict_epsilon = self.diffusion_fn(x=y, img=img, idx=idx)
                mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (
                        y - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)

                y = mu_theta_xt + torch.sqrt(1 - self.alphas[idx]) * noise

            yield y

    @torch.no_grad()
    def conditional_sampling(self, img, only_final=False):
        sampling_list = []

        final = None
        self.diffusion_fn.eval()
        for sample in self._conditional_one_diffusion_step(img):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else sampling_list

    def cond_fn(self, images, images_adv, mode):
        images = images.detach().requires_grad_(True)
        with torch.enable_grad():
            if mode == 'SSIM':
                ssim = SSIM(win_size=11, data_range=1, size_average=True, channel=3)
                logits = ssim(images, images_adv)
            elif mode == 'MSE':
                logits = F.mse_loss(images, images_adv, reduction='none')
            return torch.autograd.grad(logits.sum(), images)[0]

    def _one_diffusion_step(self, x, mode='SSIM'):
        img = x
        total_step = len(self.alpha_bars)//5
        for _ in range(2):
            self.diffusion_fn.eval()
            x = torch.sqrt(self.alpha_bars[total_step]) * x + torch.sqrt(1 - self.alpha_bars[total_step]) * torch.randn_like(x)
            for idx in reversed(range(total_step)):
                noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
                sqrt_tilde_beta = torch.sqrt((1 - self.alpha_prev_bars[idx]) / (1 - self.alpha_bars[idx]) * self.betas[idx])
                predict_epsilon = self.diffusion_fn(x=x, img=img, idx=idx)
                mu_theta_xt = torch.sqrt(1 / self.alphas[idx]) * (
                            x - self.betas[idx] / torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon)

                x_tilde = torch.sqrt(self.alpha_bars[idx]) * img + torch.sqrt(1 - self.alpha_bars[idx]) * noise

                if mode == 'SSIM':
                    a = 255/1
                    s_t = 3 * torch.sqrt(1 - self.alpha_bars[idx]) * a / (torch.sqrt(self.alpha_bars[idx]))
                    x = mu_theta_xt + sqrt_tilde_beta * noise + self.cond_fn(x, x_tilde, 'SSIM')*s_t * sqrt_tilde_beta
                elif mode == 'MSE':
                    a = 0.05
                    s_t = 3*torch.sqrt(1 - self.alpha_bars[idx])*a/(torch.sqrt(self.alpha_bars[idx]))
                    x = mu_theta_xt + sqrt_tilde_beta * noise - self.cond_fn(x, x_tilde, 'MSE') * s_t * sqrt_tilde_beta

            yield x

    @torch.no_grad()
    def distance_guidance_sampling(self, img, only_final=False):
        sampling_list = []

        final = None
        self.diffusion_fn.eval()
        for sample in self._one_diffusion_step(img):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else sampling_list
