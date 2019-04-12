from transformer import Encoder
import torch

te = Encoder.TransformerEncoder(1000,6)
a = torch.LongTensor([[1,2,3,4,4,0],[5,6,7,8,0,0]])

print(te.forward(a,torch.IntTensor([5,4])).shape)