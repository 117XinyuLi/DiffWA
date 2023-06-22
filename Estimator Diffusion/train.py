import torchvision
import numpy as np
from Diffusion_model import Diffusion_model
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
    model = Diffusion_model(device, beta_1, beta_T, T).to(device)
    if os.path.exists('model_cond.pth'):
        model.load_state_dict(torch.load('model_cond.pth'))
        print('conditional model loaded')

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
    hidden.encoder_decoder.eval()
    print('Hidden model loaded')

    if os.path.exists('code.txt'):
        with open('code.txt', 'r') as f:
            message = f.read()
            message = torch.tensor([int(i) for i in message]).float().to(device).view(16, 256)
            print('Loaded message from file')
    else:
        raise Exception('No message file found')

    # Training
    while current_iteration != total_iteration:
        try:
            data = next(dataiterator)
        except:
            dataiterator = iter(dataloader)
            data = next(dataiterator)

        data = data[0].to(device=device)

        model.train()
        encoded_data = hidden.encoder_decoder.encoder(data, message)
        loss = model.loss_fn(x=data, img=encoded_data)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()

        losses.update(loss.item())
        progress.display(current_iteration)
        current_iteration += 1
        scheduler.step()

        if current_iteration % display_iteration == 0:
            torch.save(model.state_dict(), 'model_cond.pth')
            losses.reset()


if __name__ == '__main__':
    main()
