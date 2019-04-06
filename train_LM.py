from dataloader import LMDataset
from model import DecoderRNN
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

import sys


def train_LM(lmloader, model, optimizer, criterion, pad_id, max_epoch):
    
    for epoch in range(max_epoch):

        for n, batch in enumerate(lmloader):
            optimizer.zero_grad()
            loss = 0
            input_sentences = batch['sentence']
            if torch.cuda.is_available():
                input_sentences = input_sentences.cuda()
            decoder_output, _, _ = model(input_sentences, teacher_forcing_ratio=1)
            decoder_output_reshaped = torch.cat([decoder_output[i].unsqueeze(1) for i in range(len(decoder_output))],1)
            vocab_size = decoder_output_reshaped.shape[2]
            decoder_output_reshaped = decoder_output_reshaped.view(-1, vocab_size)
            input_sentences = input_sentences[:,1:].contiguous().view(-1)
            loss = criterion(decoder_output_reshaped, input_sentences)
            loss.backward()
            optimizer.step()
            if n%10 == 0:
                print("Epoch {}, batch {}: loss = {}".format(epoch, n, loss.item()))
        PATH = 'LMcheckpoint'
        torch.save(model.state_dict(), PATH)

def sampleSentence(model, lmloader, rev_vocab):
    sample_input = next(iter(lmloader))['sentence']
    input_sentence = []
    for i in range(sample_input.shape[1]):
        input_sentence.append(rev_vocab[sample_input[0,i].item()])
    print(input_sentence)
    if torch.cuda.is_available():
        sample_input = sample_input.cuda()
    with torch.no_grad():
        _,_, out = model(sample_input,teacher_forcing_ratio=0)
    sentence = []
    for word in out['sequence']:
        sentence.append(rev_vocab[word.item()])
    print(' '.join(sentence))

def loadCheckpoint(PATH, model):
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

def main():
    # load vocab Data here!

    with open('VocabData.pkl','rb') as f:
        VocabData = pickle.load(f)
    with open('FullImageCaps.pkl','rb') as f:
        FullImageCaps = pickle.load(f)
    
    
    recovery = sys.argv[2]
    mode = sys.argv[1]

    lmdata = LMDataset(VocabData, FullImageCaps)
    lmloader = lmdata.getLoader(batchSize=128,shuffle=True)
    testloader = lmdata.getLoader(batchSize=1,shuffle=False)
    embedding = torch.Tensor(lmdata.embedding)
    vocab_size = len(lmdata.wordDict)
    max_len = 100
    hidden_size = 1024
    embedding_size = 300
    max_epoch = 20
    sos_id = lmdata.sos_id
    eos_id = lmdata.eos_id
    pad_id = lmdata.pad_id

    wordDict = VocabData['word_dict']
    rev_vocab = ['']*vocab_size
    for word in wordDict:
        rev_vocab[wordDict[word]] = word
    
    PATH = 'LMcheckpoint'

    
    model = DecoderRNN(vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id, embedding=embedding, rnn_cell='lstm')
    if recovery=='1':
        model = loadCheckpoint(PATH, model)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    if mode == 'train':
        train_LM(lmloader, model, optimizer, criterion, pad_id, max_epoch)

    sampleSentence(model, testloader, rev_vocab)


if __name__ == "__main__":
    main()