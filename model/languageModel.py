from .LSTMDecoder import DecoderRNN
from torch.nn.utils.rnn import pad_sequence

import torch.optim as optim
import torch.nn as nn

class LanguageModelLoss(nn.Module):
	"""Calculate loss based on both input sequences and outputs of decoder.
	
	Attributes:
		LanguageModelLoss(): Calculate crossentropy loss with only the outputs, using two pre-trained language models.
			The transferd style of the sample will be used to choose which language model to use.
		ReconstructLoss(): Calculate reconstruction loss of output without tranfering style.
	"""
	def __init__(self, model):
		super(LanguageModelLoss, self).__init__()
        self.model = model
        self.pad_id = pad_id
        self.criterion = nn.CrossEntropyLoss()
			

	def forward(self, outputs, lengths):
        loss = 0

        out_reshaped = torch.cat([outputs[i].unsqueeze(1) for i in range(len(outputs))],1)
        batch_size = out_reshaped.shape[0]
        
        lm_output, _, _ = model(out_reshaped, teacher_forcing_ratio=1)
        lm_output_reshape = torch.cat([lm_output[i].unsqueeze(1) for i in range(len(lm_output))],1)
        for i in range(batch_size):
            if lengths[i] <= 1:
                loss += 0
            else:
                loss += self.criterion(out_reshaped[i,1:,:],lm_output[i,:lengths[i]-1,:])

        return loss


def train_LM(model, optimizer, criterion, pad_id):
    

    for epoch in range(max_epoch):
        ld = iter(data)
        for batch in data:
            sentences = []
            for i in range(len(batch)):
                sentences.append(batch['full']['caption'])
            input = pad_sequence(sentences, batch_first=True, padding_value=pad_id)

            optimizer.zero_grad()
            loss = 0
            decoder_output, _, _ = model(input, teacher_forcing_ratio=1)
            for i in range(len(sentences)):
                length = sentences[i].shape[0] # need to check shape in debug
                for j in range(length-1):
                    loss += criterion(decoder_output[j][i],sentences[i][j+1])

            loss = criterion(decoder_output, sentences)
            loss.backward()
            optimizer.step()
        
def main():
    embedding = None
    vocab_size = None
    max_len = None
    hidden_size = None
    sos_id = None
    eos_id = None
    pad_id = None

    model = DecoderRNN(vocab_size, max_len, hidden_size, sos_id, eos_id, embedding=embedding)
    optimizer = optim.Adam()
    criterion = nn.CrossEntropyLoss()
    train_LM(model, optimizer, criterion, pad_id)