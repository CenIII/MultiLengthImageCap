import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, hiddenSize=1024):
        super(LinearModel, self).__init__()
        self.conv1 = nn.Conv2d(512, hiddenSize, (10, 9), (4, 6))
        self.conv2 = nn.Conv2d(512, hiddenSize, 5 ,stride=1)

    def forward(self, box_feat, global_feat):
        B,M = box_feat.size()[:2]
        box_feat = box_feat.view(-1,512,7,7)
        box_feat = self.conv2(box_feat).view(B,M,512,7,7)

        global_vec = []
        for i in range(len(global_feat)):
            global_vec.append(self.conv1(global_feat[i]).view(hiddenSize,-1).max(dim=1))
        global_vec = torch.cat(global_vec,dim=0)
        return box_feat, global_vec #global_feat, global_hidden,


# if __name__ == "__main__":
#     global_feat = torch.randn(512, 34, 45)
#     box_feat = torch.randn(75, 512, 7, 7)
#     image_pair = [global_feat, box_feat]

#     lm = LinearModel()
#     gf, gh, bf = lm(image_pair)
#     print(gf.shape, gh.shape, bf.shape)
