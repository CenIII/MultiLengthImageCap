import torch.nn as nn
import torch

class CrossEntropyLoss(nn.Module):

    def __init__(self, model):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels, mask):
        loss = 0
        batch_size, _, vocab_size = outputs.shape
        

        return loss/batch_size