import torch
from model.LSTMEncoder import EncoderRNN
from dataloader.dataloader import LoaderEncTrain
from model.LinearModel import LinearModel
from crit.SimilarityLoss import SimilarityLoss

def getLengths(caps):
	batchSize = len(caps)
	lengths = np.zeros(batchSize)
	for i in range(batchSize):
		cap = caps[i]
		lengths[i] = np.argmax(cap==0.)
	return lengths

# load sample data

loader = LoaderEncTrain()
data, itr, numiters = loader.getBatch()

box_feats = torch.tensor(data['box_feats'])
glob_feat = torch.tensor(data['glob_feat'])

# load linear model, transform feature tensor to semantic space
linNet = LinearModel(hiddenSize=1024)

# load LSTM encoder
lstmEnc = EncoderRNN()

# load crit 
crit = SimilarityLoss()


# output1 output2 fed into Similarity loss
out1 = linNet(box_feats, glob_feat)

out2 = lstmEnc(data['box_captions'])

capLens = getLengths(data['box_captions'])
loss = crit(out1, out2, capLens)

# loss.backward()
loss.backward()