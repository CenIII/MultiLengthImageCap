import torch
import tqdm
import numpy as np
from model.LSTMEncoder import EncoderRNN
from dataloader.dataloader import LoaderEncTrain
from model.LinearModel import LinearModel
from crit.SimilarityLoss import SimilarityLoss
import pickle
import time

def getLengths(caps):
	batchSize = len(caps)
	lengths = torch.zeros(batchSize,dtype=torch.int32)
	for i in range(batchSize):
		cap = caps[i]
		lengths[i] = torch.argmax(cap==0.)
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

if torch.cuda.is_available():
	linNet = linNet.cuda()
	lstmEnc = lstmEnc.cuda()
	crit = crit.cuda()


optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, lstmEnc.parameters()))+list(linNet.parameters()), 0.0001)


while True:
	qdar = tqdm.tqdm(range(numiters-1), total= numiters-1, ascii=True)
	for i in qdar:
		data, itr, _ = loader.getBatch()

		box_feats = torch.tensor(data['box_feats'])
		glob_feat = torch.tensor(data['glob_feat'])
		box_captions =  torch.LongTensor(data['box_captions_gt'])
		capLens = getLengths(box_captions)
		if torch.cuda.is_available():
			box_feats = box_feats.cuda()
			glob_feat = glob_feat.cuda()
			box_captions = box_captions.cuda()
			capLens = capLens.cuda()

		# output1 output2 fed into Similarity loss  # todo: incorporate glob feat
		start = time.time()
		out1 = linNet(box_feats, glob_feat)[2].unsqueeze(1)
		out2 = lstmEnc(box_captions)[0]
		end1 = time.time()
		# print('model forward: '+str(end1-start)+'s')
		# print('calc loss')
		loss = crit(out1, out2, capLens)
		end2 = time.time()
		# print('crit forward: '+str(end2-end1)+'s')
		# print('backward')
		optimizer.zero_grad()
		# loss.backward()
		loss.backward()
		optimizer.step()
		end3 = time.time()
		# print('backward: '+str(end3-end2)+'s')
		qdar.set_postfix(loss=str(np.round(loss.data.cpu().numpy(),3)))

	models = {}
	models['linNet'] = linNet.state_dict()
	models['lstmEnc'] = lstmEnc.state_dict()
	torch.save(models,'lstmEnc.pt')













