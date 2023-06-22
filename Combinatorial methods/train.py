import torchvision
import numpy as np
from Diffusion_model import Diffusion_model
from DDIM_process import DiffusionProcessDDIM
from Estimater import ResNet34
from utils import AverageMeter, ProgressMeter
import torch
import random
import os

from options import Configuration
from Hidden_model.hidden import Hidden
from noise_layers.noiser import Noiser

from config import *

random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0')


def main():
    model1 = Diffusion_model(device, beta_1, beta_T, T1).to(device)
    process = DiffusionProcessDDIM(beta_1, beta_T, T1, model1, device, shape, eta)
    if os.path.exists('model_cond1.pth'):
        model1.load_state_dict(torch.load('model_cond1.pth'))
        print('conditional model loaded')
    else:
        raise Exception('No model found')

    estimater = ResNet34().to(device)
    if os.path.exists('estimater.pth'):
        estimater.load_state_dict(torch.load('estimater.pth'))
        print('estimater loaded')
    else:
        raise Exception('No estimater found')

    model2 = Diffusion_model(device, beta_1, beta_T, T2).to(device)
    if os.path.exists('model_cond2.pth'):
        model2.load_state_dict(torch.load('model_cond2.pth'))
        print('conditional model loaded')
    optim = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    estimater.eval()

    total_iteration = 5000
    current_iteration = 0
    display_iteration = 1000

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=total_iteration, T_mult=1)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((shape[-2], shape[-1])),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=3)
    dataiterator = iter(dataloader)

    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(total_iteration, [losses], prefix='Iteration ')

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
    print('Hidden model loaded')
    hidden.encoder_decoder.eval()

    if os.path.exists('code.txt'):
        with open('code.txt', 'r') as f:
            message = f.read()
            message = torch.tensor([int(i) for i in message]).float().to(device).view(16, 256)
            print('Loaded message from file')
    else:
        raise Exception('No message file found')

    # Training
    model1.eval()
    while current_iteration != total_iteration:
        try:
            data = next(dataiterator)
        except:
            dataiterator = iter(dataloader)
            data = next(dataiterator)

        data = data[0].to(device=device)

        model2.train()
        encoded_data = hidden.encoder_decoder.encoder(data, message)
        y_N = estimater(encoded_data)
        recon = process.N_distance_guidance_sampling(encoded_data, y_N, only_final=True)
        loss = model2.loss_fn(x=data, img=recon)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1)
        optim.step()

        losses.update(loss.item())
        progress.display(current_iteration)
        current_iteration += 1
        scheduler.step()

        if current_iteration % display_iteration == 0:
            torch.save(model2.state_dict(), 'model_cond2.pth')
            losses.reset()


if __name__ == '__main__':
    main()
