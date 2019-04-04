import torch
import tqdm
import numpy as np
from model.LSTMEncoder import EncoderRNN
from dataloader.dataloader import LoaderEncTrain
from model.LinearModel import LinearModel
from crit.SimilarityLoss import SimilarityLoss
import pickle

def getLengths(caps):
	batchSize = len(caps)
	lengths = np.zeros(batchSize, dtype=np.int32)
	for i in range(batchSize):
		cap = caps[i]
		lengths[i] = int(np.argmax(cap==0.))
	return lengths

# load sample data

loader = LoaderEncTrain()
data, itr, numiters = loader.getBatch()
numiters = int(numiters)

# load vocab data
with open('./data/VocabData.pkl', 'rb') as f:
	VocabData = pickle.load(f)

# load linear model, transform feature tensor to semantic space
linNet = LinearModel(hiddenSize=1024)
# load LSTM encoder
lstmEnc = EncoderRNN(len(VocabData['word_dict']), 15, 1024, 300,
                 input_dropout_p=0, dropout_p=0,
                 n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=False,
                 embedding_parameter=VocabData['word_embs'], update_embedding=False)
# load crit 
crit = SimilarityLoss(1,1,1)

optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, lstmEnc.parameters()))+list(linNet.parameters()), 0.001)


qdar = tqdm.tqdm(range(numiters), total= numiters, ascii=True)
for i in qdar:
	data, itr, numiters = loader.getBatch()

	box_feats = torch.tensor(data['box_feats'])
	glob_feat = torch.tensor(data['glob_feat'])
	box_captions =  torch.LongTensor(data['box_captions_gt'])

	# output1 output2 fed into Similarity loss  # todo: incorporate glob feat
	out1 = linNet(box_feats, glob_feat)[2].unsqueeze(1)
	out2 = lstmEnc(box_captions)[0]

	capLens = getLengths(box_captions)
	print('calc loss')
	loss = crit(out1, out2, capLens)
	print('backward')
	optimizer.zero_grad()
	# loss.backward()
	loss.backward()
	optimizer.step()
	qdar.set_postfix(loss=str(np.round(loss.data,3)))














