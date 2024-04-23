import pywt
import torch
import torchvision

from options import Configuration
from Hidden_model.hidden import Hidden
from noise_layers.noiser import Noiser

from Bit_Accuracy import Bit_Accuracy

import os
import random
import numpy as np
import cv2

seed = 999
random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
shape = (3, 32, 32)
device = torch.device('cuda:0')


def main():

    if not os.path.exists('images'):
        os.mkdir('images')

    cifar10 = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.Resize((shape[-2], shape[-1])),
                                               torchvision.transforms.ToTensor()
                                           ]))
    cifar10_loader = torch.utils.data.DataLoader(cifar10, batch_size=1, shuffle=False)

    if os.path.exists('code.txt'):
        with open('code.txt', 'r') as f:
            message = f.read()
            message = torch.tensor([int(i) for i in message]).float().to(device).view(16, 256)[0:1]
            print('Loaded message from file')
    else:
        raise Exception('No message file found')

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

    bit_acc_without_ca = 0
    bit_acc_without_ch = 0
    bit_acc_without_cv = 0
    bit_acc_without_cd = 0

    # pic = next(iter(cifar10_loader))[0].to(device)
    print('Start testing')
    print(len(cifar10_loader))
    iteration = 0

    for pic in cifar10_loader:
        iteration += 1
        print('iteration: ', iteration)
        pic = pic[0].to(device)
        # 保存原始图片
        piv_cv = cv2.cvtColor(pic[0].detach().transpose(0, 1).transpose(1, 2).cpu().numpy()*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite('original_pic.png', piv_cv)

        encoded_pic = hidden.encoder_decoder.encoder(pic, message.view(-1, 256))

        # 保存隐藏信息后的图片
        piv_cv = cv2.cvtColor(encoded_pic[0].detach().permute(1, 2, 0).cpu().numpy()*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite('images/encoded_pic.png', piv_cv)

        # 将图片转换为灰度图片并保存
        gray_pic = cv2.cvtColor(piv_cv, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('images/gray_pic.png', gray_pic)

        # 将图片转换为小波变换后的图片并保存
        coeffs = pywt.dwt2(gray_pic, 'haar')
        ca, (ch, cv, cd) = coeffs
        cv2.imwrite('images/ca.png', ca)
        cv2.imwrite('images/ch.png', ch)
        cv2.imwrite('images/cv.png', cv)
        cv2.imwrite('images/cd.png', cd)

        # 将原图删除ca
        coeffs = (np.zeros_like(ca), (ch, cv, cd))
        pic_without_ca = pywt.idwt2(coeffs, 'haar')
        pic_without_ca = cv2.cvtColor(pic_without_ca, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('images/pic_without_ca.png', cv2.cvtColor(pic_without_ca, cv2.COLOR_RGB2BGR))
        pic_without_ca = torch.from_numpy(pic_without_ca).permute(2, 0, 1).float().to(device) / 255
        pic_without_ca = pic_without_ca.unsqueeze(0)
        recon_message = hidden.encoder_decoder.decoder(pic_without_ca)
        bit_accuracy = Bit_Accuracy()
        bit_acc_without_ca += bit_accuracy(recon_message, message.view(-1, 256))
        print('bit accuracy without ca: ', bit_accuracy(recon_message, message.view(-1, 256)))

        # 将原图删除ch
        coeffs = (ca, (np.zeros_like(ch), cv, cd))
        pic_without_ch = pywt.idwt2(coeffs, 'haar')
        pic_without_ch = cv2.cvtColor(pic_without_ch, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('images/pic_without_ch.png', cv2.cvtColor(pic_without_ch, cv2.COLOR_RGB2BGR))
        pic_without_ch = torch.from_numpy(pic_without_ch).permute(2, 0, 1).float().to(device) / 255
        pic_without_ch = pic_without_ch.unsqueeze(0)
        recon_message = hidden.encoder_decoder.decoder(pic_without_ch)
        bit_accuracy = Bit_Accuracy()
        bit_acc_without_ch += bit_accuracy(recon_message, message.view(-1, 256))
        print('bit accuracy without ch: ', bit_accuracy(recon_message, message.view(-1, 256)))

        # 将原图删除cv
        coeffs = (ca, (ch, np.zeros_like(cv), cd))
        pic_without_cv = pywt.idwt2(coeffs, 'haar')
        pic_without_cv = cv2.cvtColor(pic_without_cv, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('images/pic_without_cv.png', cv2.cvtColor(pic_without_cv, cv2.COLOR_RGB2BGR))
        pic_without_cv = torch.from_numpy(pic_without_cv).permute(2, 0, 1).float().to(device) / 255
        pic_without_cv = pic_without_cv.unsqueeze(0)
        recon_message = hidden.encoder_decoder.decoder(pic_without_cv)
        bit_accuracy = Bit_Accuracy()
        bit_acc_without_cv += bit_accuracy(recon_message, message.view(-1, 256))
        print('bit accuracy without cv: ', bit_accuracy(recon_message, message.view(-1, 256)))

        # 将原图删除cd
        coeffs = (ca, (ch, cv, np.zeros_like(cd)))
        pic_without_cd = pywt.idwt2(coeffs, 'haar')
        pic_without_cd = cv2.cvtColor(pic_without_cd, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('images/pic_without_cd.png', cv2.cvtColor(pic_without_cd, cv2.COLOR_RGB2BGR))
        pic_without_cd = torch.from_numpy(pic_without_cd).permute(2, 0, 1).float().to(device) / 255
        pic_without_cd = pic_without_cd.unsqueeze(0)
        recon_message = hidden.encoder_decoder.decoder(pic_without_cd)
        bit_accuracy = Bit_Accuracy()
        bit_acc_without_cd += bit_accuracy(recon_message, message.view(-1, 256))
        print('bit accuracy without cd: ', bit_accuracy(recon_message, message.view(-1, 256)))

    print('bit accuracy without ca: ', bit_acc_without_ca/iteration)
    print('bit accuracy without ch: ', bit_acc_without_ch/iteration)
    print('bit accuracy without cv: ', bit_acc_without_cv/iteration)
    print('bit accuracy without cd: ', bit_acc_without_cd/iteration)


if __name__ == '__main__':
    main()
