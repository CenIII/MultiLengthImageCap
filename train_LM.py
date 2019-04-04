from dataloader import LMDataset
from model import DecoderRNN
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

def train_LM(lmloader, model, optimizer, criterion, pad_id, max_epoch):
    
    for epoch in range(max_epoch):

        for n, batch in enumerate(lmloader):
            optimizer.zero_grad()
            loss = 0
            if torch.cuda.is_available():
                batch = batch.cuda()
            input_sentences = batch['sentence']
            lengths = batch['lengths']
            batch_size = lengths.shape[0]
            decoder_output, _, _ = model(input_sentences, teacher_forcing_ratio=1)
            decoder_output_reshaped = torch.cat([decoder_output[i].unsqueeze(1) for i in range(len(decoder_output))],1)
            for i in range(batch_size):
                loss += criterion(decoder_output_reshaped[i,:lengths[i]-1], input_sentences[i,1:lengths[i]])
            loss /= batch_size
            loss.backward()
            optimizer.step()
            if n%10 == 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, loss.item()))

def main():
    # load vocab Data here!

    with open('VocabData.pkl','rb') as f:
        VocabData = pickle.load(f)
    with open('FullImageCaps.pkl','rb') as f:
        FullImageCaps = pickle.load(f)
        
    lmdata = LMDataset(VocabData, FullImageCaps)
    lmloader = lmdata.getLoader(batchSize=128,shuffle=True)

    embedding = torch.Tensor(lmdata.embedding)
    vocab_size = len(lmdata.wordDict)
    max_len = 100
    hidden_size = 1024
    embedding_size = 300
    max_epoch = 10
    sos_id = lmdata.sos_id
    eos_id = lmdata.eos_id
    pad_id = lmdata.pad_id

    model = DecoderRNN(vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id, embedding=embedding, rnn_cell='lstm')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_LM(lmloader, model, optimizer, criterion, pad_id, max_epoch)

if __name__ == "__main__":
    main()