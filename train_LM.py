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
            input_sentences = batch['sentence']
            if torch.cuda.is_available():
                input_sentences = input_sentences.cuda()
            batch_size = input_sentences.shape[0]
            decoder_output, _, _ = model(input_sentences, teacher_forcing_ratio=1)
            decoder_output_reshaped = torch.cat([decoder_output[i].unsqueeze(1) for i in range(len(decoder_output))],1)
            vocab_size = decoder_output_reshaped.shape[2]
            decoder_output_reshaped = decoder_output_reshaped.view(-1, vocab_size)
            input_sentences = input_sentences[:,1:].view(-1)
            loss = criterion(decoder_output_reshaped, input_sentences[:,1:])
            loss.backward()
            optimizer.step()
            if n%10 == 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, loss.item()))
        PATH = 'LMcheckpoint'
        torch.save(model.state_dict(), PATH)

def sampleSentence(model, sos_id):
    sample_input = torch.LongTensor([[sos_id]])
    _,_, out = model(sample_input,teacher_forcing_ratio=0)
    print(out['sequence'])

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
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    train_LM(lmloader, model, optimizer, criterion, pad_id, max_epoch)

if __name__ == "__main__":
    main()