import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LinearModel(nn.Module):
    def __init__(self, hiddenSize=1024):
        super(LinearModel, self).__init__()
        self.conv1 = nn.Conv2d(512, hiddenSize, 7, stride=2)
        self.conv2 = nn.Conv2d(512, hiddenSize, 5 ,stride=1)
        self.hiddenSize = hiddenSize

    def forward(self, box_feat, global_feat): # box_feat [8, 2, 512, 7, 7], globel_feat list of [512, 26, 45]
        B,M = box_feat.size()[:2]
        box_feat = box_feat.view(-1,512,7,7)
        box_feat = self.conv2(box_feat).view(B,M,self.hiddenSize,3,3)

        global_vec = []
        for i in range(len(global_feat)):
            global_vec.append(self.conv1(global_feat[i].unsqueeze(0)).view(self.hiddenSize,-1).max(dim=1)[0])
        global_vec = torch.stack(global_vec,dim=0)

        global_hidden = global_vec.unsqueeze(0)
        encoder_hidden = (global_hidden,torch.zeros_like(global_hidden).to(device))
        B,M,D,H,W = box_feat.size()
        encoder_outputs = box_feat.permute(0,1,3,4,2).contiguous().view(B,-1,D)

        return encoder_outputs, encoder_hidden # box_feat [8,2,hiddensize,3,3], global_vec [8, hidden_size]


# if __name__ == "__main__":
#     global_feat = torch.randn(512, 34, 45)
#     box_feat = torch.randn(75, 512, 7, 7)
#     image_pair = [global_feat, box_feat]

#     lm = LinearModel()
#     gf, gh, bf = lm(image_pair)
#     print(gf.shape, gh.shape, bf.shape)
