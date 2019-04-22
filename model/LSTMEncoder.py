import torch.nn as nn
import torch

from .baseRNN import BaseRNN
import time

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)
        embedding_parameter (torch.Tensor, optional): Pre-trained embedding.  The size of the tensor has to match
            the size of the embedding parameter: (vocab_size, hidden_size).  The embedding layer would be initialized
            with the tensor if provided (default: None).
        update_embedding (bool, optional): If the embedding should be updated during training (default: False).

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)

    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, hidden_size, embedding_size,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=True,
                 embedding_parameter=None, update_embedding=False):

        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.max_len = max_len
        self.variable_lengths = variable_lengths
        self.linear = nn.Linear(vocab_size, embedding_size, bias=False)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding_parameter is not None:
            embedding_parameter = torch.FloatTensor(embedding_parameter)
            self.linear.weight = nn.Parameter(embedding_parameter.transpose(1,0))
            self.embedding.weight = nn.Parameter(embedding_parameter)
        self.embedding.weight.requires_grad = update_embedding
        self.linear.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(embedding_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def forward(self, input_var, use_prob_vector=False, input_lengths=None,max_len=15):
        """
        Applies an RNN encoder to a batch of input sequences.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
                   or (batch, seq_len, vocab_size): if use_prob_vector = True
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
            use_prob_vector (bool, optional): if use probability vector instead of index vector of word (default: False)

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
        """
        
        if use_prob_vector:
            embedded = self.linear(input_var)
        else:
            embedded = self.embedding(input_var)

        input_lengths, indices = torch.sort(input_lengths, descending=True, out=None) #sort for pack_padded
        embedded = embedded[indices,:,:]

        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)

        self.rnn.flatten_parameters()
        output, _ = self.rnn(embedded)

        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=max_len)

            output_sort_back = torch.zeros_like(output) #sort back to match labels
            for i in range(output.shape[0]):
                output_sort_back[indices[i]] = output[i]
            output = output_sort_back

        return output