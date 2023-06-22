import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 1.0
        return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


if __name__ == '__main__':
    transformer = transforms.ToTensor()
    psnr = PSNR()
    img1 = Image.open('DDPM_results/display5.0/original.png')
    img2 = Image.open('DDPM_results/display5.0/encoded.png')
    img3 = Image.open('DDPM_results/display5.0/reconstructed.png')
    img1 = transformer(img1)
    img2 = transformer(img2)
    img3 = transformer(img3)
    print('PSNR(original recon): ', psnr(img1, img3).item())
    print('PSNR(original encoded): ', psnr(img1, img2).item())
