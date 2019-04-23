from transformer import Encoder
from LSTMEncoder import EncoderRNN
import torch

te = Encoder.TransformerEncoder(1000,6)

a = torch.LongTensor([[1,2,3,4,0,0],[10,0,0,0,0,0],[5,6,7,8,9,0]])
#print(te.forward(a,torch.IntTensor([5,4])).shape)

ls = EncoderRNN(1000, 15, 4096, 300)
print(ls.forward(a,use_prob_vector=False, input_lengths=torch.Tensor([4,1,5])).shape)
