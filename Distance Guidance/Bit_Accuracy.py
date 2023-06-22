import torch
import torch.nn as nn


class Bit_Accuracy(nn.Module):
    def __init__(self):
        super(Bit_Accuracy, self).__init__()

    def forward(self, input_img, target, threshold=0.5):
        input_img = input_img.view(-1).sigmoid()
        target = target.view(-1)
        input_over_threshold = input_img > threshold
        target_over_threshold = target > threshold
        correct = torch.sum(input_over_threshold == target_over_threshold)

        return correct / input_img.size(0)

