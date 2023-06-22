import torch
from Diffusion_model import Diffusion_model
from torch.nn import functional as F
import numpy as np
from SSIM import SSIM


class DiffusionProcessDDIM():
    def __init__(self, beta_1, beta_T, T, diffusion_fn: Diffusion_model, device, shape, eta, tau=1, encode_ratio=1, scheduling='uniform'):
        '''
        beta_1        : beta_1 of diffusion process
        beta_T        : beta_T of diffusion process
        T             : step of diffusion process
        diffusion_fn  : trained diffusion network
        shape         : data shape
        eta           : coefficient of sigma
        tau           : accelerating of diffusion process
        encode_ratio  : ratio of encoding
        scheduling    : scheduling mode of diffusion process
        '''
        beta_T = beta_T - (beta_T - beta_1) * (1 - encode_ratio)
        T = int(T * encode_ratio)
        self.T = T
        self.encode_ratio = encode_ratio
        self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
        self.alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), self.alpha_bars[:-1]])

        self.shape = shape
        self.sigmas = torch.sqrt((1 - self.alpha_prev_bars) / (1 - self.alpha_bars)) * torch.sqrt(
            1 - (self.alpha_bars / self.alpha_prev_bars))
        self.diffusion_fn = diffusion_fn
        self.device = device
        self.eta = eta
        self.tau = tau
        self.scheduling = scheduling

    def _get_process_scheduling(self, reverse=True):
        if self.scheduling == 'uniform':
            diffusion_process = list(range(0, len(self.alpha_bars), self.tau)) + [len(self.alpha_bars) - 1]
        elif self.scheduling == 'exp':
            diffusion_process = (np.linspace(0, np.sqrt(len(self.alpha_bars) * 0.8), self.tau) ** 2)
            diffusion_process = [int(s) for s in list(diffusion_process)] + [len(self.alpha_bars) - 1]
        else:
            assert 'Not Implementation'

        diffusion_process = zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])) if reverse else zip(
            diffusion_process[1:], diffusion_process[:-1])
        return diffusion_process

    def _one_reverse_diffusion_step(self, img):
        total_steps = len(self.alpha_bars)
        for i in range(3):
            diffusion_process = list(range(0, total_steps, self.tau)) + [total_steps - 1]
            y = torch.sqrt(self.alpha_bars[total_steps-1]) * img + torch.randn_like(img) * torch.sqrt(1 - self.alpha_bars[total_steps-1])
            for prev_idx, idx in zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])):
                self.diffusion_fn.eval()
                noise = torch.zeros_like(y) if idx == 0 else torch.randn_like(y)
                predict_epsilon = self.diffusion_fn(x=y, img=img, idx=idx)
                sigma = self.sigmas[idx] * self.eta

                predicted_x0 = torch.sqrt(self.alpha_bars[prev_idx]) * (
                            y - torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_bars[idx])
                direction_pointing_to_xt = torch.sqrt(1 - self.alpha_bars[prev_idx] - sigma ** 2) * predict_epsilon
                y = predicted_x0 + direction_pointing_to_xt + sigma * noise

            total_steps = int(total_steps//1.5)

            yield y

    @torch.no_grad()
    def conditional_sampling(self, images, only_final=False):
        sampling_list = []

        final = None
        for sample in self._one_reverse_diffusion_step(images):
            final = sample
            if not only_final:
                sampling_list.append(final)

        return final if only_final else sampling_list

    @torch.no_grad()
    def probabilityflow(self, x, reverse, reverse_data=None, mask_M=None, threshold=0.5):
        '''
        reverse : if True, backward(noise -> data) else forward(data -> noise)
        '''

        def reparameterize_sigma(idx):
            return torch.sqrt((1 - self.alpha_bars[idx]) / self.alpha_bars[idx])

        def reparameterize_x(x, idx):
            return x / torch.sqrt(self.alpha_bars[idx])

        if mask_M is None:
            mask_M = torch.ones([x.size(0), self.T, x.size(2), x.size(3)]).to(device=self.device)
        else:
            mask_M = (mask_M > threshold).float()

        self.diffusion_fn.eval()

        diffusion_process = self._get_process_scheduling(reverse=reverse)
        return_data = []
        for idx_delta_t, idx in diffusion_process:
            x_bar_delta_t = reparameterize_x(x, idx) + 0.5 * (
                        reparameterize_sigma(idx_delta_t) ** 2 - reparameterize_sigma(idx) ** 2) / reparameterize_sigma(
                idx) * self.diffusion_fn(x, idx)
            x = x_bar_delta_t * torch.sqrt(self.alpha_bars[idx_delta_t])

            if not reverse:
                return_data.append(x)

            else:
                x = mask_M[:, idx-1:idx, :, :] * x + (1 - mask_M[:, idx-1:idx, :, :]) * reverse_data.pop()

        return return_data if not reverse else x

    def cond_fn(self, images, images_adv, mode):
        images = images.detach().requires_grad_(True)
        with torch.enable_grad():
            if mode == 'MSE':
                logits = F.mse_loss(images, images_adv, reduction='none')
            elif mode == 'SSIM':
                ssim_loss = SSIM(win_size=11, data_range=1, size_average=True, channel=3)
                logits = -ssim_loss(images, images_adv)
            logits = torch.log(1-torch.tanh(logits))
            return torch.autograd.grad(logits.sum(), images)[0]

    def _conditional_reverse_diffusion_step(self, x, mode='SSIM'):
        img = x.clone()
        total_steps = len(self.alpha_bars)//10
        for _ in range(2):
            self.diffusion_fn.eval()
            x = torch.sqrt(self.alpha_bars[total_steps-1]) * x + torch.sqrt(
                1 - self.alpha_bars[total_steps-1]) * torch.randn_like(x)
            diffusion_process = list(range(0, total_steps, self.tau)) + [total_steps - 1]
            for prev_idx, idx in zip(reversed(diffusion_process[:-1]), reversed(diffusion_process[1:])):
                self.diffusion_fn.eval()
                noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)

                x_tilde = torch.sqrt(self.alpha_bars[idx]) * img + torch.sqrt(1 - self.alpha_bars[idx]) * noise
                if mode == 'MSE':
                    a = 1
                elif mode == 'SSIM':
                    a = 255/0.01
                s_t = 3 * torch.sqrt(1 - self.alpha_bars[idx]) * a / (torch.sqrt(self.alpha_bars[idx]))
                predict_epsilon = self.diffusion_fn(x=x, img=img, idx=idx) + self.cond_fn(x_tilde, x, mode) * s_t * torch.sqrt(1-self.alpha_bars[idx])

                sigma = self.sigmas[idx] * self.eta

                predicted_x0 = torch.sqrt(self.alpha_bars[prev_idx]) * (
                        x - torch.sqrt(1 - self.alpha_bars[idx]) * predict_epsilon) / torch.sqrt(self.alpha_bars[idx])
                direction_pointing_to_xt = torch.sqrt(1 - self.alpha_bars[prev_idx] - sigma ** 2) * predict_epsilon
                x = predicted_x0 + direction_pointing_to_xt + sigma * noise

            yield x

    @torch.no_grad()
    def distance_guidance_sampling(self, images, only_final=False):

        self.diffusion_fn.eval()
        final = None
        final_list = []
        for sample in self._conditional_reverse_diffusion_step(images):
            final = sample
            if not only_final:
                final_list.append(final)

        return final if only_final else final_list




