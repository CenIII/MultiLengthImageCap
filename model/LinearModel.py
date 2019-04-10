import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, hiddenSize=1024):
        super(LinearModel, self).__init__()
        # self.conv1 = nn.Conv2d(512, hiddenSize, (10, 9), (4, 6))
        # self.bn1 = nn.BatchNorm2d(hiddenSize)
        self.conv2 = nn.Conv2d(512, hiddenSize, 3, 3)
        self.bn2 = nn.BatchNorm2d(hiddenSize)

    def forward(self, box_feat):#, global_feat):
        # global_feat = global_feat.unsqueeze(dim=0)
        # global_feat = self.bn1(self.conv1(global_feat))
        # global_hidden = F.avg_pool2d(global_feat, 7).squeeze()
        box_feat = self.bn2(self.conv2(box_feat)).unsqueeze(1)#self.bn2(self.conv2(box_feat))
        return box_feat #global_feat, global_hidden,


# if __name__ == "__main__":
#     global_feat = torch.randn(512, 34, 45)
#     box_feat = torch.randn(75, 512, 7, 7)
#     image_pair = [global_feat, box_feat]

#     lm = LinearModel()
#     gf, gh, bf = lm(image_pair)
#     print(gf.shape, gh.shape, bf.shape)