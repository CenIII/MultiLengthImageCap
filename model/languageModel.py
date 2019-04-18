from .LSTMDecoder import DecoderRNN
from torch.nn.utils.rnn import pad_sequence


import torch.nn as nn
import torch

class LanguageModelLoss(nn.Module):

    def __init__(self, PATH, vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id, use_prob_vector=False):
        super(LanguageModelLoss, self).__init__()
        model = DecoderRNN(vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id, rnn_cell='lstm', use_prob_vector=use_prob_vector)
        self.model = self.loadCheckpoint(PATH, model)

    def loadCheckpoint(self, PATH, model):
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model

    def criterion(self, decoder_out, lm_out, mask=None):
        N = decoder_out.shape[0]
        _loss = torch.mul(torch.log(decoder_out), lm_out)
        if mask is not None:
            _loss = torch.mul(_loss, mask)
        return -torch.sum(_loss)/N

    def forward(self, outputs, lengths=None):
        loss = 0

        out_reshaped = torch.cat([outputs[i].unsqueeze(1) for i in range(len(outputs))],1)
        N, T, vocab_size  = out_reshaped.shape
        
        lm_output, _, _ = self.model(out_reshaped, teacher_forcing_ratio=1)
        lm_output_reshape = torch.cat([lm_output[i].unsqueeze(1) for i in range(len(lm_output))],1)
        out_reshaped = out_reshaped[:,1:,:].contiguous().view(-1, vocab_size)
        lm_output_reshape = lm_output_reshape[:,:-1,:].contiguous().view(-1, vocab_size)
        
        mask = None
        if lengths is not None:
            mask = torch.zeros(N, T)
            for i in range(len(lengths.shape[0])):
                mask[i,:lengths[i]] += 1
        mask = mask[:,1:].contiguous().view(-1, 1)
        loss = self.criterion(out_reshaped,lm_output_reshape, mask)

        return loss



        
