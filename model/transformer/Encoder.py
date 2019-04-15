''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec, hidden_size,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,
            embedding_parameter=None ,update_embedding=False):

        super().__init__()

        self.len_max_seq = len_max_seq
        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        if embedding_parameter:
            self.src_word_emb.weight = nn.Parameter(embedding_parameter)
        self.src_word_emb.weight.requires_grad = update_embedding

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        

    def forward(self, src_seq, len_src_seq, return_attns=False):

        enc_slf_attn_list = []
        # -- create src_pos
        batch_size = len_src_seq.shape[0]
        src_pos = torch.zeros((batch_size, self.len_max_seq), dtype=torch.long)
        for i in range(batch_size):
            nums = torch.arange(1, len_src_seq[i] + 1, dtype=torch.long)
            zeros = torch.zeros((self.len_max_seq - len_src_seq[i]).item(), dtype=torch.long)
            src_pos[i] = torch.cat((nums, zeros))

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output

class TransformerEncoder(nn.Module):
    r"""
    Applies a Transformer Encoder to a batch of input sequence.

    Args:
        n_src_vocab (int): size of the vocabulary
        len_max_seq (int): a maximum allowed length for the sequence to be processed
        d_word_vec (int): word embedding size
        hidden_size (int): the number of output features
        d_inner (int): size of inner feed forward layer
        n_layers (int): number of stacked encoder layers
        n_head (int): number of heads
        d_k (int): dimension of query and key
        d_v (int): dimension of value
        dropout (float): dropout parameter
        embedding_parameter (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **src_seq**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **len_src_seq** list of int: list that contains the lengths of sequences in the mini-batch

    Outputs: output
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence

    Examples::

         >>> encoder = TransformerEncoder(n_src_vocab, len_max_seq, d_word_vec, hidden_size)
         >>> output = encoder(src_seq, len_src_seq)

    """

    def __init__(
            self,
            n_src_vocab, len_max_seq, d_word_vec=512, hidden_size=1024, 
            d_inner=2048, n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1,
            embedding_parameter=None, update_embedding=False):
        super().__init__()

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq, d_word_vec=d_word_vec,
            hidden_size=hidden_size, d_model=d_word_vec, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
            embedding_parameter=embedding_parameter, update_embedding=update_embedding)
        
        self.project_hidden = nn.Linear(d_word_vec, hidden_size)

        #assert d_model == d_word_vec, \
        #'To facilitate the residual connections, \
        # the dimensions of all module outputs shall be the same.'

    def forward(self, src_seq, len_src_seq):
        enc_output = self.encoder(src_seq, len_src_seq)
        enc_output = self.project_hidden(enc_output)
        return enc_output
