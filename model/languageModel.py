from .LSTMDecoderLM import DecoderRNN
from torch.nn.utils.rnn import pad_sequence


import torch.nn as nn
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SymbolDecoder(object):
    """docstring for SymbolDecoder"""
    def __init__(self,word_dict):
        super(SymbolDecoder, self).__init__()
        self.word_dict = word_dict
        self.ind2word = self.makeInd2Word()
    
    def makeInd2Word(self):
        ind2word = {}
        for k,v in self.word_dict.items():
            ind2word[v]=k
        return ind2word
    
    def decode(self,ind_seq):
        ret = []
        if isinstance(ind_seq,list):
            for seq in ind_seq:
                ret.append(self.decode(seq))
            return ret
        else:
            return self.ind2word[int(ind_seq)]

class LanguageModelLoss(nn.Module):

    def __init__(self, wordDict, PATH, vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id, use_prob_vector=False):
        super(LanguageModelLoss, self).__init__()
        model = DecoderRNN(vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id, rnn_cell='lstm', beamSearchMode=True)
        self.model = self.loadCheckpoint(PATH, model)
        self.symdec = SymbolDecoder(wordDict)

    def loadCheckpoint(self, model_path, model):
        pt = torch.load(model_path)

        def subload(model, pt_dict):
            model_dict = model.state_dict()
            pretrained_dict = {}
            for k, v in pt_dict.items():
                if (k in model_dict):
                    pretrained_dict[k] = v #if ('embedding.weight' not in k) else v.transpose(1,0)
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            return model

        model = subload(model, pt)
        pt = None
        return model

    # def loadCheckpoint(self, PATH, model):
    #     model.load_state_dict(torch.load(PATH))
    #     model.eval()
    #     return model



    def criterion(self, decoder_out, lm_out, mask=None):
        N = decoder_out.shape[0]
        _loss = torch.mul(torch.log(lm_out), decoder_out)
        if mask is not None:
            _loss = torch.mul(_loss, mask)
        return -torch.sum(_loss)/N
    # def length_to_mask(self, length, max_len=None, dtype=None):
    #     """length: B.
    #     return B x max_len.
    #     If max_len is None, then max of length will be used.
    #     """
    #     length = length.to(device)
    #     assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    #     max_len = max_len or length.max().item()
    #     mask = torch.arange(max_len, device=device).expand(len(length), max_len) < length.unsqueeze(1)
    #     if dtype is not None:
    #         mask = torch.as_tensor(mask, dtype=dtype, device=device)
    #     return mask

    def probVec2Symbols(self,probvec):
        sequence_symbols = []
        for i in range(len(probvec)):
            symbols = probvec[i].topk(1)[1]
            sequence_symbols.append(symbols)
        wordSeq = self.symdec.decode(sequence_symbols)
        return wordSeq

    def forward(self, outputs, lengths=None, beamStates=None, max_len=15, verbose=False):  # [8, 15, 10878]
        loss = 0

        # out_reshaped = outputs# torch.cat([outputs[i].unsqueeze(1) for i in range(len(outputs))],1)
        # N, T, vocab_size  = out_reshaped.shape
        # out_top1 = out_reshaped.topk(1)[1].squeeze()
        K,N,_,V = beamStates['probVec'][2].shape
        T = len(beamStates['probVec'])-1
        lm_output, _, ret_dict = self.model(inputs=None, beamStates=beamStates, max_len=max_len)

        lm_output_reshape = torch.cat([lm_output[i].unsqueeze(1) for i in range(len(lm_output))],1)
        out_reshaped = out_reshaped[:,1:,:].contiguous().view(-1, vocab_size)
        lm_output_reshape = lm_output_reshape.contiguous().view(-1, vocab_size)
        
        mask = None
        if lengths is not None:
            mask = torch.zeros(N, T).to(device)
            for i in range(len(lengths)):
                mask[i,:lengths[i]] += 1
        
        # mask = self.length_to_mask(lengths,dtype=torch.float)
        mask = mask[:,1:].contiguous().view(-1, 1)

        # decode outputs
        if verbose:
            print('lm outputs: '+str(self.probVec2Symbols(lm_output_reshape)))
            print('dec outputs: '+str(self.probVec2Symbols(out_reshaped)))

        loss = self.criterion(out_reshaped,lm_output_reshape, mask)

        return loss



        
