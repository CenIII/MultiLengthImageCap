import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, encHiddenSize=4096, decHiddenSize=1024):
        super(LinearModel, self).__init__()
        self.conv1 = nn.Conv2d(encHiddenSize, decHiddenSize, 1, stride=1)
        self.conv2 = nn.Conv2d(512, encHiddenSize, 5 ,stride=1)
        self.encHiddenSize = encHiddenSize
        self.decHiddenSize = decHiddenSize

    def forward(self, box_feat, global_feat): # box_feat [8, 2, 512, 7, 7], globel_feat list of [512, 26, 45]
        B,M = box_feat.size()[:2]
        box_feat = box_feat.view(-1,512,7,7)
        box_feat = self.conv2(box_feat)

        # global_vec = []
        # for i in range(len(global_feat)):
        #     global_vec.append(self.conv1(global_feat[i].unsqueeze(0)).view(self.hiddenSize,-1).max(dim=1)[0])
        # global_vec = torch.stack(global_vec,dim=0)

        box_feat_dec = self.conv1(box_feat).view(B,M,self.decHiddenSize,3,3)
        box_feat = box_feat.view(B,M,self.encHiddenSize,3,3)
        global_vec = box_feat_dec.permute(0,2,1,3,4).contiguous().view(B,self.decHiddenSize,-1).max(dim=2)[0]
        return box_feat, box_feat_dec, global_vec # box_feat [8,2,hiddensize,3,3], global_vec [8, hidden_size]


# if __name__ == "__main__":
#     global_feat = torch.randn(512, 34, 45)
#     box_feat = torch.randn(75, 512, 7, 7)
#     image_pair = [global_feat, box_feat]

#     lm = LinearModel()
#     gf, gh, bf = lm(image_pair)
#     print(gf.shape, gh.shape, bf.shape)

class LinearModelEnc(nn.Module):
    def __init__(self, hiddenSize=1024):
        super(LinearModelEnc, self).__init__()
        # self.conv1 = nn.Conv2d(512, hiddenSize, (10, 9), (4, 6))
        # self.bn1 = nn.BatchNorm2d(hiddenSize)
        self.conv2 = nn.Conv2d(512, hiddenSize, 4, 4)
        # self.bn2 = nn.BatchNorm2d(hiddenSize)

    def forward(self, box_feat):#, global_feat):
        # global_feat = global_feat.unsqueeze(dim=0)
        # global_feat = self.bn1(self.conv1(global_feat))
        # global_hidden = F.avg_pool2d(global_feat, 7).squeeze()
        box_feat = self.conv2(box_feat).unsqueeze(1)#self.bn2(self.conv2(box_feat))
        return box_feat #global_feat, global_hidden,