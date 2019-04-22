from dataloader import LMDataset
from model import DecoderRNN
from model import LM_DecoderRNN
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

import sys
import json
import string

from model import LanguageModelLoss

def train_LM(lmloader, model, optimizer, criterion, pad_id, max_epoch, max_len):
    
    for epoch in range(max_epoch):

        for n, batch in enumerate(lmloader):
            optimizer.zero_grad()
            loss = 0
            input_sentences = batch['sentence']
            if torch.cuda.is_available():
                input_sentences = input_sentences.cuda()
            
            decoder_output, _, _ = model(input_sentences, teacher_forcing_ratio=0, max_len=max_len)
            decoder_output_reshaped = torch.cat([decoder_output[i].unsqueeze(1) for i in range(len(decoder_output))],1)
            decoder_output = None
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
    ld = iter(lmloader)
    for i in range(10):
        print('Sample sentence {}:'.format(i))
        sample_input = next(ld)['sentence']
        input_sentence = []
        for i in range(sample_input.shape[1]):
            input_sentence.append(rev_vocab[sample_input[0,i].item()])
        print(' '.join(input_sentence))
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()
        with torch.no_grad():
            _,_, out = model(sample_input,teacher_forcing_ratio=1)
        sentence = []
        for word in out['sequence']:
            sentence.append(rev_vocab[word.item()])
        print(' '.join(sentence))

def loadCheckpoint(PATH, model):
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

def loadData(PATH):
    with open(PATH, 'r') as f:
        data = json.load(f)

    sentences = []
    for d in data:
        s = d['paragraph'].strip().split('.')[:-1]
        for sentence in s:
            sentences.append(words_preprocess(sentence))
    return sentences

def words_preprocess(phrase):
  """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
  replacements = {
    '½': 'half',
    '—' : '-',
    '™': '',
    '¢': 'cent',
    'ç': 'c',
    'û': 'u',
    'é': 'e',
    '°': ' degree',
    'è': 'e',
    '…': '',
  }
  for k, v in replacements.items():
    phrase = phrase.replace(k, v)
  return str(phrase).lower().translate(str.maketrans('','',string.punctuation)).split()

def main():
    # load vocab Data here!

    with open('VocabData.pkl','rb') as f:
        VocabData = pickle.load(f)
    # with open('FullImageCaps.pkl','rb') as f:
    #     FullImageCaps = pickle.load(f)
    FullImageCaps = loadData("full_image_descriptions.json")
    
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
    max_epoch = 40
    sos_id = lmdata.sos_id
    eos_id = lmdata.eos_id
    pad_id = lmdata.pad_id

    wordDict = VocabData['word_dict']
    rev_vocab = ['']*vocab_size
    for word in wordDict:
        rev_vocab[wordDict[word]] = word
    
    they = torch.zeros(1, vocab_size)
    are = torch.zeros(1, vocab_size)
    students = torch.zeros(1, vocab_size)
    _from = torch.zeros(1, vocab_size)
    that = torch.zeros(1, vocab_size)
    school = torch.zeros(1, vocab_size)
    they_id = wordDict['they']
    are_id = wordDict['are']
    students_id = wordDict['students']
    from_id = wordDict['from']
    that_id = wordDict['that']
    school_id = wordDict['school']

    they[0,they_id] = 1
    are[0,are_id] = 1
    students[0,students_id] = 1
    _from[0,from_id]=1
    that[0,that_id]=1
    school[0,school_id]=1

    strange_sentence = torch.cat([they, are, are, are, are, are], 0).unsqueeze(0)
    regular_sentence = torch.cat([they, are, students, _from, that, school], 0).unsqueeze(0)

    PATH = 'LMcheckpoint'

    
    model = LM_DecoderRNN(vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id, embedding=embedding, rnn_cell='lstm')
    if recovery=='1':
        model = loadCheckpoint(PATH, model)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.NLLLoss(ignore_index=pad_id)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    if mode == 'train':
        train_LM(lmloader, model, optimizer, criterion, pad_id, max_epoch, max_len)

    if mode == 'test':
        lm_loss = LanguageModelLoss(PATH, vocab_size, max_len, hidden_size, embedding_size, sos_id, eos_id, use_prob_vector=True)
        loss1 = lm_loss(strange_sentence)
        loss2 = lm_loss(regular_sentence)
        print(loss1.item(), loss2.item())

    sampleSentence(model, testloader, rev_vocab)


if __name__ == "__main__":
    main()