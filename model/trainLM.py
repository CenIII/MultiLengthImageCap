from languageModel import *
import torch.nn as nn
import torch.optim as optim

def main():
    data = None
    embedding = None
    vocab_size = None
    max_len = None
    hidden_size = None
    sos_id = None
    eos_id = None
    pad_id = None

    model = DecoderRNN(vocab_size, max_len, hidden_size, sos_id, eos_id, embedding=embedding, rnn_cell='lstm')
    optimizer = optim.Adam()
    criterion = nn.CrossEntropyLoss()
    train_LM(data, model, optimizer, criterion, pad_id)

if __name__ == "__main__":
    main()