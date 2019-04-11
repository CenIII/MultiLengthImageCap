import torch
import torch.nn as nn
import tqdm
import numpy as np
from model.LSTMEncoder import EncoderRNN
from dataloader.dataloader import LoaderEnc
from model.LinearModel import LinearModel
from crit.SimilarityLoss import SimilarityLoss
import pickle
import time
import sys
import argparse
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def getLengths(caps):
	batchSize = len(caps)
	lengths = torch.zeros(batchSize,dtype=torch.int32)
	for i in range(batchSize):
		cap = caps[i]
		lengths[i] = (cap==0).nonzero()[0][0]
	return lengths

def reloadModel(model_path,linNet,lstmEnc):
	pt = torch.load(model_path)

	def subload(model,pt_dict):
		model_dict = model.state_dict()
		pretrained_dict = {}
		for k, v in pt_dict.items():
			if(k[7:] in model_dict):
				pretrained_dict[k[7:]] = v
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		# 3. load the new state dict
		model.load_state_dict(model_dict)
		return model

	linNet = subload(linNet,pt['linNet'])
	lstmEnc = subload(lstmEnc,pt['lstmEnc'])
	
	return linNet,lstmEnc

def train(loader,linNet,lstmEnc,crit,optimizer,savepath, batchImgs=4):
	os.makedirs(savepath,exist_ok=True)
	# if torch.cuda.is_available():
	linNet = nn.DataParallel(linNet,device_ids=[0, 1]).to(device)
	lstmEnc = nn.DataParallel(lstmEnc,device_ids=[0, 1]).to(device)
	crit = crit.to(device)
	data, itr, numiters = loader.getBatch()
	numiters = int(int(numiters)/batchImgs)
	epoch = 0
	# loss_epoch_list = []
	logger = open(os.path.join(savepath,'loss_history'),'w')

	def loadMultiImgData(loader,numImgs=4):
		box_feats = []
		box_captions = []
		capLens = []
		for i in range(numImgs):
			data, itr, _ = loader.getBatch()
			box_feats.append(torch.tensor(data['box_feats']).to(device))
			box_captions.append(torch.LongTensor(data['box_captions_gt']).to(device))
			capLens.append(getLengths(box_captions[-1]).to(device))
		box_feats = torch.cat(box_feats)
		box_captions = torch.cat(box_captions)
		capLens = torch.cat(capLens)

		# sort decreasing order
		inds = torch.argsort(-capLens)
		box_feats = box_feats[inds]
		box_captions = box_captions[inds]
		capLens = capLens[inds]

		return box_feats, box_captions, capLens


	while True:
		qdar = tqdm.tqdm(range(numiters-1), total= numiters-1, ascii=True)
		loss_itr_list = []
		for i in qdar:
			box_feats, box_captions, capLens = loadMultiImgData(loader,numImgs=batchImgs)
			
			# output1 output2 fed into Similarity loss  # todo: incorporate glob feat
			out1 = linNet(box_feats)
			out2 = lstmEnc(box_captions,input_lengths=capLens)
			
			loss = crit(out1, out2, capLens)
			loss_itr_list.append(loss.data.cpu().numpy())
			
			optimizer.zero_grad()
			
			loss.backward()
			optimizer.step()
			
			qdar.set_postfix(loss=str(np.round(loss.data.cpu().numpy(),3)))

		loss_epoch_mean = np.mean(loss_itr_list)
		print('epoch '+str(epoch)+' mean loss:'+str(np.round(loss_epoch_mean,5)))
		# loss_epoch_list.append(loss_epoch_mean)
		logger.write(str(np.round(loss_epoch_mean,5))+'\n')
		logger.flush()
		models = {}
		models['linNet'] = linNet.state_dict()
		models['lstmEnc'] = lstmEnc.state_dict()
		torch.save(models,os.path.join(savepath ,'lstmEnc.pt'))
		epoch += 1


def eval(loader,linNet,lstmEnc,crit):
	# for now evaluation means to do similarity matrix check.
	linNet = linNet.to(device)
	lstmEnc = lstmEnc.to(device)
	linNet.eval()
	lstmEnc.eval()

	data, itr, _ = loader.getBatch()

	box_feats = torch.tensor(data['box_feats']).to(device)
	# glob_feat = torch.tensor(data['glob_feat'])
	box_captions =  torch.LongTensor(data['box_captions_gt']).to(device)
	capLens = getLengths(box_captions).to(device)
	# sort decreasing order
	inds = torch.argsort(-capLens)
	box_feats = box_feats[inds]
	box_captions = box_captions[inds]
	capLens = capLens[inds]

	# check the similarity loss based on argument
	out1 = linNet(box_feats)
	out2 = lstmEnc(box_captions,input_lengths=capLens)
	Similarity_matrix = crit.generate_similarity_matrix(out1, out2, capLens)
	torch.save(Similarity_matrix, "similarity_matrix")

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-e','--evaluate_mode',
		action='store_true',
	  	help='check similarity matrix.')
	parser.add_argument('-p','--model_path',
		default='./lstmEnc.pt')
	parser.add_argument('-s','--save_path',
		default='./save/default/')
	parser.add_argument('-b','--batch_imgs',
		default=4, type=int)
	args = parser.parse_args()
	return args

if __name__ == '__main__':

	args = parseArgs()

	# load vocab data
	with open('./data/VocabData.pkl', 'rb') as f:
		VocabData = pickle.load(f)

	# load linear model, transform feature tensor to semantic space
	linNet = LinearModel(hiddenSize=1024)
	# load LSTM encoder
	lstmEnc = EncoderRNN(len(VocabData['word_dict']), 15, 1024, 300,
	                 input_dropout_p=0, dropout_p=0,
	                 n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=True,
	                 embedding_parameter=VocabData['word_embs'], update_embedding=False)
	# load crit
	crit = SimilarityLoss(0.5,0.5,1)

	if args.evaluate_mode:			# evaluation mode
		loader = LoaderEnc(mode='test')
		linNet,lstmEnc = reloadModel(args.model_path,linNet,lstmEnc)
		eval(loader,linNet,lstmEnc,crit)
	else:							# train mode
		optimizer = torch.optim.Adam(list(filter(lambda p: p.requires_grad, lstmEnc.parameters()))+list(linNet.parameters()), 0.001)
		loader = LoaderEnc()
		train(loader,linNet,lstmEnc,crit,optimizer,args.save_path,batchImgs=args.batch_imgs)













