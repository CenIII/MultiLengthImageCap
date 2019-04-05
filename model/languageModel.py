from .LSTMDecoder import DecoderRNN
from torch.nn.utils.rnn import pad_sequence


import torch.nn as nn
import torch

class LanguageModelLoss(nn.Module):

    def __init__(self, model):
        super(LanguageModelLoss, self).__init__()
        self.model = model

    def criterion(self, decoder_out, lm_out, mask=None):
        N = decoder_out.shape[0]
        _loss = torch.mul(torch.log(decoeder_out), lm_out)
        if mask is not None:
            _loss = torch.mul(_loss, mask)
        return -torch.sum(_loss)/N

    def forward(self, outputs, lengths, mask=None):
        loss = 0

        out_reshaped = torch.cat([outputs[i].unsqueeze(1) for i in range(len(outputs))],1)
        vocab_size  = out_reshaped.shape[2]
        
        lm_output, _, _ = self.model(out_reshaped, teacher_forcing_ratio=1)
        lm_output_reshape = torch.cat([lm_output[i].unsqueeze(1) for i in range(len(lm_output))],1)
        out_reshaped = out_reshaped[:,1:,:].contiguous().view(-1, vocab_size)
        lm_output_reshape = lm_output_reshape[:,:-1,:].contiguous().view(-1, vocab_size)
        loss = self.criterion(out_reshaped,lm_output_reshape, mask)

        return loss


# def train_LM(data, model, optimizer, criterion, pad_id):
    

#     for epoch in range(max_epoch):
#         ld = iter(data)
#         for batch in data:
#             sentences = []
#             for i in range(len(batch)):
#                 sentences.append(batch['full']['caption'])
#             input = pad_sequence(sentences, batch_first=True, padding_value=pad_id)

#             optimizer.zero_grad()
#             loss = 0
#             decoder_output, _, _ = model(input, teacher_forcing_ratio=1)
#             for i in range(len(sentences)):
#                 length = sentences[i].shape[0] # need to check shape in debug
#                 for j in range(length-1):
#                     loss += criterion(decoder_output[j][i],sentences[i][j+1])

#             loss = criterion(decoder_output, sentences)
#             loss.backward()
#             optimizer.step()
        
