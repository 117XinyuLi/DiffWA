import os
import torch
import torchvision
from torchvision.utils import save_image

from options import *
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_layers.crop import Crop
from noise_layers.cropout import Cropout
from noise_layers.resize import Resize
from noise_layers.dropout import Dropout
from noise_layers.quantization import Quantization
from noise_layers.jpeg_compression import JpegCompression

import random
import numpy as np

seed = 999
random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(model, train_options, dataset, device):
    for epoch in range(train_options.start_epoch, train_options.number_of_epochs + 1):
        for batch_idx, (data, _) in enumerate(dataset[0]):
            data = data.to(device)

            model.train()
            losses, pics = model.train_on_batch([data, dataset[1]])

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(dataset[0].dataset)} Loss: {losses["loss"]:.6f}]')

                messages = pics[2].view(-1, 1, 16, 16)

                save_image(data, f'images/pic_original.png', nrow=8)
                save_image(pics[0], f'images/pic_encoded.png', nrow=8)
                save_image(pics[1], f'images/pic_noised.png', nrow=8)
                save_image(messages, f'images/message{epoch}.png', nrow=8)

                torch.save(model.state_dict(), f'checkpoints_hidden.pth')


def main():
    start_epoch = 1
    train_options = TrainingOptions(
        batch_size=16,
        number_of_epochs=100,
        start_epoch=start_epoch)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CIFAR10 = torchvision.datasets.CIFAR10(root='CIFAR10', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                           ]))

    cifar10_loader = torch.utils.data.DataLoader(CIFAR10, batch_size=train_options.batch_size, shuffle=True,
                                                 num_workers=2, drop_last=True)


    if os.path.exists('code.txt'):
        with open('code.txt', 'r') as f:
            message = f.read()
            message = torch.tensor([int(i) for i in message]).float().to(device).view(train_options.batch_size, 256)
            print('Loaded message from file')
    else:
        message = torch.randint(0, 2, (train_options.batch_size*256,)).float().to(device)
        with open('code.txt', 'w') as f:
            f.write(''.join([str(int(i)) for i in message]))
            message = message.view(train_options.batch_size, 256)
            print('Created new message')

    dataset = [cifar10_loader, message]

    crop = Crop([0.9, 1.0], [0.9, 1.0])
    cropout = Cropout([0.9, 1.0], [0.9, 1.0])
    resize = Resize([0.8, 0.9])
    dropout = Dropout([0.8, 0.9])
    quantization = Quantization(device)
    jpeg_compression = JpegCompression(device)
    noise_config = [crop, cropout, resize, dropout, quantization, jpeg_compression]

    hidden_config = HiDDenConfiguration(H=32, W=32,
                                        message_length=256,
                                        encoder_blocks=10, encoder_channels=256,
                                        decoder_blocks=7, decoder_channels=128,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        decoder_loss=1,
                                        encoder_loss=0.7,
                                        adversarial_loss=1e-3,
                                        )

    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser)

    if os.path.exists('checkpoints_hidden.pth'):
        model.load_state_dict(torch.load('checkpoints_hidden.pth'))
        print('Loaded model from checkpoint')

    train(model, train_options, dataset, device)


if __name__ == '__main__':
    main()
