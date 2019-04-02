from dataloader import LMDataset
import torch.nn as nn
import torch.optim as optim

def train_LM(lmloader, model, optimizer, criterion, pad_id):
    
    for epoch in range(max_epoch):
        ld = iter(lmloader)
        for batch in ld:
            optimizer.zero_grad()
            loss = 0
            decoder_output, _, _ = model(batch, teacher_forcing_ratio=1)

            loss = criterion(decoder_output, batch)
            loss.backward()
            optimizer.step()

def main():
    # load vocab Data here!

    with open('VocabData.pkl','rb') as f:
        VocabData = pickle.load(f)
    with open('FullImageCaps.pkl','rb') as f:
        VocabData = pickle.load(f)
        
    lmdata = LMDataset(VocabData, FullImageCaps)
    lmloader = lmdata.getLoader(batchSize=128,shuffle=True)

    embedding = lmdata.embedding
    vocab_size = lmdata.vocab_size
    max_len = 100
    hidden_size = 1024
    sos_id = lmdata.sos_id
    eos_id = lmdata.eos_id
    pad_id = lmdata.pad_id

    model = DecoderRNN(vocab_size, max_len, hidden_size, sos_id, eos_id, embedding=embedding, rnn_cell='lstm')
    optimizer = optim.Adam()
    criterion = nn.CrossEntropyLoss()
    train_LM(lmloader, model, optimizer, criterion, pad_id)

if __name__ == "__main__":
    main()