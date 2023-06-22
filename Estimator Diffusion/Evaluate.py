import torchvision
import numpy as np
from DDPM_process import DiffusionProcessDDPM
from DDIM_process import DiffusionProcessDDIM
from Diffusion_model import Diffusion_model
import torch
import random
import os
from torchvision.utils import save_image

from SSIM import SSIM
from Bit_Accuracy import Bit_Accuracy

from options import Configuration
from Hidden_model.hidden import Hidden
from noise_layers.noiser import Noiser

from PSNR import PSNR
from config import *

from Estimater import ResNet34

random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0')


def main():

    model = Diffusion_model(device, beta_1, beta_T, T).to(device=device)
    if os.path.exists('model_cond.pth'):
        model.load_state_dict(torch.load('model_cond.pth'))
        print('conditional model loaded')
    else:
        raise Exception('No model found')

    if not os.path.exists('Evaluation'):
        os.mkdir('Evaluation')

    if eva_mode == 'DDIM':
        if not os.path.exists('Evaluation/DDIM'):
            os.mkdir('Evaluation/DDIM')
    elif eva_mode == 'DDPM':
        if not os.path.exists('Evaluation/DDPM'):
            os.mkdir('Evaluation/DDPM')
    else:
        raise ValueError('eva_mode must be DDIM or DDPM')

    save_dir = 'Evaluation/' + eva_mode

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((shape[-2], shape[-1])),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, drop_last=True, num_workers=2, shuffle=False)
    dataiterator = iter(dataloader)

    data = next(dataiterator)[0].to(device)
    if eva_mode == 'DDIM':
        process = DiffusionProcessDDIM(beta_1, beta_T, T, model, device, shape, eta)
    elif eva_mode == 'DDPM':
        process = DiffusionProcessDDPM(beta_1, beta_T, T, model, device, shape)

    estimater = ResNet34().to(device)
    if os.path.exists('estimater.pth'):
        estimater.load_state_dict(torch.load('estimater.pth'))
        print('estimater loaded')
    else:
        raise Exception('No estimater found')

    hidden_config = Configuration(W=shape[1], H=shape[2],
                                  message_length=256,
                                  encoder_blocks=10, encoder_channels=256,
                                  decoder_blocks=7, decoder_channels=128,
                                  use_discriminator=True,
                                  discriminator_blocks=3, discriminator_channels=64,
                                  encoder_loss=1,
                                  decoder_loss=0.7,
                                  adversarial_loss=1e-3,
                                  )

    noise_config = []
    noiser = Noiser(noise_config, device)
    hidden = Hidden(hidden_config, device, noiser).to(device)

    hidden.load_state_dict(torch.load('checkpoints_hidden.pth'))
    hidden.eval()
    estimater.eval()

    if os.path.exists('code.txt'):
        with open('code.txt', 'r') as f:
            message = f.read()
            message = torch.tensor([int(i) for i in message]).float().to(device).view(16, 256)
            print('Loaded message from file')
    else:
        raise Exception('No message file found')

    encoded = hidden.encoder_decoder.encoder(data, message.view(-1, 256))
    y_N = estimater(encoded)
    if not use_distance_guidance:
        data_reconstructed = process.N_conditional_sampling(encoded, y_N, True)
    else:
        data_reconstructed = process.N_distance_guidance_sampling(encoded, y_N, True)
    del estimater, model, process, y_N
    save_image(data, save_dir + '/original.png', nrow=4)
    save_image(encoded, save_dir + '/encoded.png', nrow=4)
    save_image(data_reconstructed, save_dir + '/reconstructed.png', nrow=4)

    difference = (data_reconstructed - data).abs()
    difference = (difference[:, 0, :, :] + difference[:, 1, :, :] + difference[:, 2, :, :])/3
    difference = torch.where(difference > 0.02, difference, torch.zeros_like(difference))
    difference = (difference*15).clip(0, 1).unsqueeze(1)
    save_image(difference, save_dir + '/difference.png', nrow=4)

    ssim = SSIM(win_size=11, data_range=1, size_average=True, channel=3)
    print('SSIM(original recon): ', ssim(data, data_reconstructed).item())
    print('SSIM(encoded recon): ', ssim(encoded, data_reconstructed).item())
    print('SSIM(original encoded): ', ssim(data, encoded).item())

    psnr = PSNR()
    print('PSNR(original recon): ', psnr(data, data_reconstructed).item())
    print('PSNR(encoded recon): ', psnr(encoded, data_reconstructed).item())
    print('PSNR(original encoded): ', psnr(data, encoded).item())

    discrim_out = hidden.discriminator(data)
    print('data Discriminator output: ', discrim_out.sigmoid().mean().item())

    discrim_out = hidden.discriminator(encoded)
    print('encoded Discriminator output: ', discrim_out.sigmoid().mean().item())

    discrim_out = hidden.discriminator(data_reconstructed)
    print('recon Discriminator output: ', discrim_out.sigmoid().mean().item())

    encoded_decoded = hidden.encoder_decoder.decoder(encoded).view(-1, 1, 16, 16)
    save_image(encoded_decoded, save_dir + '/decoded_encoded.png', nrow=4)
    data_decoded = hidden.encoder_decoder.decoder(data).view(-1, 1, 16, 16)
    save_image(data_decoded, save_dir + '/decoded_original.png', nrow=4)
    recon_decoded = hidden.encoder_decoder.decoder(data_reconstructed).view(-1, 1, 16, 16)
    save_image(recon_decoded, save_dir + '/decoded_reconstructed.png', nrow=4)

    bit_accuracy = Bit_Accuracy()
    print('Bit accuracy(message encoded_decoded): ', bit_accuracy(encoded_decoded.view(-1, 256), message.view(-1, 256),).item())
    print('Bit accuracy(message data_decoded): ', bit_accuracy(data_decoded.view(-1, 256), message.view(-1, 256)).item())
    print('Bit accuracy(message recon_decoded): ', bit_accuracy(recon_decoded.view(-1, 256), message.view(-1, 256)).item())


if __name__ == '__main__':
    main()
