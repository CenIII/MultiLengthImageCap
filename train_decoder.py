import torch
import torch.nn as nn
import tqdm
import numpy as np
from model.LSTMEncoder import EncoderRNN
from model.LSTMDecoder import DecoderRNN
from dataloader.dataloader import LoaderDec
from model.LinearModel import LinearModel
from crit.SimilarityLoss import SimilarityLoss
from model.languageModel  import LanguageModelLoss
import pickle
import time
import sys
import argparse
import os
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def getLengths(caps):
	batchSize = len(caps)
	lengths = torch.zeros(batchSize, dtype=torch.int32)
	for i in range(batchSize):
		cap = caps[i]
		nonz = (cap == 0).nonzero()
		lengths[i] = nonz[0][0] if len(nonz) > 0 else len(cap)
	return lengths


def reloadModel(model_path, linNet, lstmEnc):
	pt = torch.load(model_path)

	def subload(model, pt_dict):
		model_dict = model.state_dict()
		pretrained_dict = {}
		for k, v in pt_dict.items():
			if (k in model_dict):
				pretrained_dict[k] = v if ('linear.weight' not in k) else v.transpose(1,0)
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		# 3. load the new state dict
		model.load_state_dict(model_dict)
		return model

	linNet = subload(linNet, pt['linNet'])
	lstmEnc = subload(lstmEnc, pt['lstmEnc'])
	pt = None
	for p in linNet.conv2.parameters():
		p.requires_grad = False

	return linNet, lstmEnc

def reloadDec(model_path, linNet, lstmDec):
	pt = torch.load(model_path)

	def subload(model, pt_dict):
		model_dict = model.state_dict()
		pretrained_dict = {}
		for k, v in pt_dict.items():
			if (k in model_dict):
				pretrained_dict[k] = v
		# 2. overwrite entries in the existing state dict
		model_dict.update(pretrained_dict)
		# 3. load the new state dict
		model.load_state_dict(model_dict)
		return model

	linNet = subload(linNet, pt['linNet'])
	lstmDec = subload(lstmDec, pt['lstmDec'])
	pt = None
	for p in linNet.conv2.parameters():
		p.requires_grad = False

	return linNet, lstmDec


def makeInp(*inps):
	ret = []
	for inp in inps:
		if isinstance(inp, list):
			ret.append(makeInp(*inp))
		elif isinstance(inp, int):
			ret.append(inp)
		else:
			ret.append(inp.to(device))
	return ret


def evaluate(loader, lstmDec, linNet, lstmEnc, LM, crit):
	# if torch.cuda.is_available():
	lstmDec = lstmDec.to(device).eval()
	linNet = linNet.to(device).eval()  # nn.DataParallel(linNet,device_ids=[0, 1]).to(device)
	lstmEnc = lstmEnc.to(device).eval()  # nn.DataParallel(lstmEnc,device_ids=[0, 1]).to(device)
	LM = LM.to(device).eval()
	crit = crit.to(device).eval()

	ld = iter(loader)
	numiters = len(ld)
	qdar = tqdm.tqdm(range(numiters), total=numiters, ascii=True)
	loss_itr_list = []
	
	with torch.no_grad(): # evaluate mode
		
		for i in qdar:

			# step 1: load data
			batchdata = next(ld)
			box_feats, box_global_feats, numBoxes = makeInp(*batchdata)  # box_feats: (numImage,numBoxes,512,7,7) box_global_feats: list, numImage [(512,34,56)]
			
			# step 2: data transform by linNet
			box_feat, global_hidden = linNet(box_feats, box_global_feats)
			
			# step 3: decode to captions by lstmDec
			encoder_hidden, encoder_outputs = linOut2DecIn(global_hidden,box_feat)
			decoder_outputs, decoder_hidden, ret_dict = lstmDec(encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs, max_len=int(5*numBoxes)) # box_feat [8, 4, 4096, 3, 3]
			
			# step 4: calculate loss
				# Loss 1: Similarity loss
			lengths = torch.LongTensor(ret_dict['length']).to(device)
			decoder_outputs = torch.stack([decoder_outputs[i] for i in range(len(decoder_outputs))], 1) # decoder_outputs [8, 15, 10878]
			encoder_outputs = lstmEnc(decoder_outputs, use_prob_vector=True, input_lengths=lengths, max_len=int(5*numBoxes))
			loss1, loss_reg = crit(box_feat, encoder_outputs, lengths) #box_feat [8, 5, 4096, 3, 3], encoder_outputs [8, 15, 4096]
				# Loss 2: LM loss
			loss2 =  LM(decoder_outputs, lengths, max_len=int(5*numBoxes))


			loss = loss1+loss_reg+loss2


			loss_itr_list.append(loss.data.cpu().numpy())

			lstmEnc.zero_grad()
			LM.zero_grad()
			optimizer.zero_grad()


			loss.backward()
			optimizer.step()







def train(loader, lstmDec, linNet, lstmEnc, LM, crit, optimizer, savepath):
	os.makedirs(savepath, exist_ok=True)
	# if torch.cuda.is_available():
	lstmDec = lstmDec.to(device)
	linNet = linNet.to(device)  # nn.DataParallel(linNet,device_ids=[0, 1]).to(device)
	lstmEnc = lstmEnc.to(device)  # nn.DataParallel(lstmEnc,device_ids=[0, 1]).to(device)
	LM = LM.to(device)
	crit = crit.to(device)

	epoch = 0
	logger = open(os.path.join(savepath, 'loss_history'), 'w')

	def saveStateDict(linNet, lstmEnc):
		models = {}
		models['linNet'] = linNet.state_dict()
		models['lstmDec'] = lstmEnc.state_dict()
		torch.save(models, os.path.join(savepath, 'lstmDec.pt'))

	def linOut2DecIn(global_hidden, box_feat):	# box_feat [8, 4, 4096, 3, 3]
		global_hidden = global_hidden.unsqueeze(0)
		encoder_hidden = (global_hidden,torch.zeros_like(global_hidden).to(device))
		B,M,D,H,W = box_feat.size()
		encoder_outputs = box_feat.permute(0,1,3,4,2).contiguous().view(B,-1,D)
		return encoder_hidden, encoder_outputs

	def lstr(ts,pres=3):
		return str(np.round(ts.data.cpu().numpy(), 3))

	while True:
		ld = iter(loader)
		numiters = len(ld)
		qdar = tqdm.tqdm(range(numiters), total=numiters, ascii=True)
		loss_itr_list = []

		for i in qdar:

			# step 1: load data
			batchdata = next(ld)
			box_feats, box_global_feats, numBoxes = makeInp(*batchdata)  # box_feats: (numImage,numBoxes,512,7,7) box_global_feats: list, numImage [(512,34,56)]
			
			# step 2: data transform by linNet
			box_feat, global_hidden = linNet(box_feats, box_global_feats)
			
			# step 3: decode to captions by lstmDec
			encoder_hidden, encoder_outputs = linOut2DecIn(global_hidden,box_feat)
			decoder_outputs, decoder_hidden, ret_dict = lstmDec(encoder_hidden=encoder_hidden, encoder_outputs=encoder_outputs, max_len=int(5*numBoxes)) # box_feat [8, 4, 4096, 3, 3]
			
			# step 4: calculate loss
				# Loss 1: Similarity loss
			lengths = torch.LongTensor(ret_dict['length']).to(device)
			decoder_outputs = torch.stack([decoder_outputs[i] for i in range(len(decoder_outputs))], 1) # decoder_outputs [8, 15, 10878]
			encoder_outputs = lstmEnc(decoder_outputs, use_prob_vector=True, input_lengths=lengths, max_len=int(5*numBoxes))
			loss1, loss_reg = crit(box_feat, encoder_outputs, lengths) #box_feat [8, 5, 4096, 3, 3], encoder_outputs [8, 15, 4096]
				# Loss 2: LM loss
			loss2 =  LM(decoder_outputs, lengths, max_len=int(5*numBoxes))


			loss = loss1+loss_reg+loss2


			loss_itr_list.append(loss.data.cpu().numpy())

			lstmEnc.zero_grad()
			LM.zero_grad()
			optimizer.zero_grad()


			loss.backward()
			optimizer.step()

			qdar.set_postfix(simiLoss=lstr(loss1),regLoss=lstr(loss_reg),lmLoss=lstr(loss2))
			if i > 0 and i % 50 == 0: #save model quickly
				saveStateDict(linNet, lstmDec)

		loss_epoch_mean = np.mean(loss_itr_list)
		print('epoch ' + str(epoch) + ' mean loss:' + str(np.round(loss_epoch_mean, 5)))
		# loss_epoch_list.append(loss_epoch_mean)
		logger.write(str(np.round(loss_epoch_mean, 5)) + '\n')
		logger.flush()
		saveStateDict(linNet, lstmDec)
		epoch += 1


def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-e', '--evaluate_mode',
						action='store_true',
						help='check similarity matrix.')
	parser.add_argument('-p', '--model_path',
						default='./lstmEnc.pt')
	parser.add_argument('-s', '--save_path',
						default='./save/default/')
	parser.add_argument('-b', '--batch_imgs',
						default=4, type=int)
	parser.add_argument('-c', '--cont_model_path',
						default='./save/default/')
	args = parser.parse_args()
	return args


if __name__ == '__main__':

	args = parseArgs()

	# load vocab data
	with open('./data/VocabData.pkl', 'rb') as f:
		VocabData = pickle.load(f)

	# load linear model, transform feature tensor to semantic space
	linNet = LinearModel(hiddenSize=4096)

	sos_id = VocabData['word_dict']['<START>']
	eos_id = VocabData['word_dict']['<END>']
	# load LSTM encoder

	lstmDec = DecoderRNN(vocab_size=len(VocabData['word_dict']),max_len=15,sos_id=sos_id, eos_id=eos_id , embedding_size=300,hidden_size=4096,
						 embedding_parameter=VocabData['word_embs'], update_embedding=False ,use_attention=True,use_prob_vector=True)

	lstmEnc = EncoderRNN(len(VocabData['word_dict']), max_len=15, hidden_size=4096, embedding_size=300,
						 input_dropout_p=0, dropout_p=0,
						 n_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=False,
						 embedding_parameter=VocabData['word_embs'], update_embedding=False)
	# todo: reload lstmEnc
	linNet,lstmEnc = reloadModel(args.model_path, linNet, lstmEnc)

	LM =  LanguageModelLoss(
		PATH="./data/LMcheckpoint", vocab_size=len(VocabData['word_dict']),max_len=15,sos_id=sos_id, eos_id=eos_id , embedding_size=300,hidden_size=1024, use_prob_vector=True
	)
	crit = SimilarityLoss(0.5, 0.5, 1)


	if args.evaluate_mode:  # evaluation mode
		if args.cont_model_path == None:
			assert(0)
		linNet,lstmDec = reloadDec(args.cont_model_path,linNet,lstmDec)
		dataset = LoaderDec()
		loader = DataLoader(dataset, batch_size=args.batch_imgs, shuffle=False, num_workers=2,
							collate_fn=dataset.collate_fn)
		evaluate(loader, lstmDec, linNet, lstmEnc, LM, crit)

	else:  # train mode
		if args.cont_model_path is not None:
			linNet,lstmDec = reloadDec(args.cont_model_path,linNet,lstmDec)
		optimizer = torch.optim.Adam(
			list(filter(lambda p: p.requires_grad, lstmDec.parameters())) + list(linNet.conv1.parameters()), 0.0001)
		dataset = LoaderDec()
		loader = DataLoader(dataset, batch_size=args.batch_imgs, shuffle=False, num_workers=2,
							collate_fn=dataset.collate_fn)
		train(loader, lstmDec, linNet, lstmEnc, LM, crit, optimizer, args.save_path)









