import torchvision
import numpy as np
from Diffusion_model import Diffusion_Model
from utils import AverageMeter, ProgressMeter
import torch
import random
import os

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

    model = Diffusion_Model(device, beta_1, beta_T, T).to(device=device)
    if os.path.exists('distance_guidance_model.pth'):
        model.load_state_dict(torch.load('distance_guidance_model.pth'))
        print('Model loaded')
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    total_iteration = 5000
    current_iteration = 0
    display_iteration = 1000

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((shape[-2], shape[-1])),
        torchvision.transforms.ToTensor()
    ])

    dataset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, drop_last=True, num_workers=3)
    dataiterator = iter(dataloader)

    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(total_iteration, [losses], prefix='Iteration ')

    # Training
    while current_iteration != total_iteration:
        try:
            data = next(dataiterator)
        except:
            dataiterator = iter(dataloader)
            data = next(dataiterator)
        data = data[0].to(device=device)
        loss = model.loss_fn(data)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss.item())
        progress.display(current_iteration)
        current_iteration += 1

        if current_iteration % display_iteration == 0:
            torch.save(model.state_dict(), 'distance_guidance_model.pth')
            losses.reset()


if __name__ == '__main__':
    main()
