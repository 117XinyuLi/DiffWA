import torch
import torchvision
from Estimater import ResNet34
from tqdm import tqdm

from options import Configuration
from Hidden_model.hidden import Hidden
from noise_layers.noiser import Noiser

from torchvision.utils import save_image
from Bit_Accuracy import Bit_Accuracy

import random
import os
import numpy as np

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
    estimater = ResNet34().to(device=device)
    alpha_bars = torch.cumprod(1 - torch.linspace(start=beta_1, end=beta_T, steps=T), dim=0).to(device=device)
    if os.path.exists('estimater.pth'):
        estimater.load_state_dict(torch.load('estimater.pth'))
        print('Estimater model loaded')

    epochs = 100
    optim = torch.optim.AdamW(estimater.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=epochs, T_mult=1)
    criterion = torch.nn.MSELoss()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((shape[-2], shape[-1])),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=3)

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

    noise = torch.randn(batch_size, shape[0], shape[1], shape[2]).to(device)
    for epoch in range(epochs):
        t = tqdm(dataloader)
        for i, (img, _) in enumerate(t):

            t.set_description(f'Epoch {epoch}')

            img = img.to(device)
            optim.zero_grad()

            encoded = hidden.encoder_decoder.encoder(img, message)
            x_N = torch.sqrt(alpha_bars[N]) * img + torch.sqrt(1 - alpha_bars[N]) * noise
            x_N_hat = estimater(encoded)
            loss = criterion(x_N_hat, x_N)

            loss.backward()
            optim.step()
            t.set_postfix(loss=loss.item())
            t.update(1)

            if i % 500 == 0:
                torch.save(estimater.state_dict(), 'estimater.pth')
                x_N_hat = estimater(x_N)
                decoded = hidden.encoder_decoder.decoder(x_N_hat)
                bit_accuracy = Bit_Accuracy()
                print(bit_accuracy(decoded, message))
                save_image(x_N_hat, f'images/x_N_hat_{i}.png')
                save_image(x_N, f'images/x_N_{i}.png')

        scheduler.step()

        torch.save(estimater.state_dict(), 'estimater.pth')


if __name__ == '__main__':
    main()

