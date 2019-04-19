# a python dataloader that loads dumped features files prodeced by lua program.
import os
import glob
# import json
import torchfile
import random
from torch.utils.data import Dataset
import torch
from utils.math import softmax
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import numpy as np

class _BaseDataLoader(Dataset):
	"""docstring for BaseDataLoader"""
	def __init__(self):
		super(_BaseDataLoader, self).__init__()
		# self.pipeIndex = 0
		# save pick_confirm for all pip indices
		self.dataPipePath = './densecap/data_pipeline/'
		self.pipeLen = 60

	# def updatePipeIndex(self):
	# 	self.pipeIndex = (self.pipeIndex+1)%self.pipeLen

	def init_pick_confirm_files(self):
		for i in range(self.pipeLen):
			filename = glob.glob(self.dataPipePath+'data_'+str(i)+'*')
			if len(filename)<1:
				save_file = os.path.join(self.dataPipePath,'pick_confirm_'+str(i))
				with open(save_file,'w') as f:
					f.write('a')

	def loadOneJson(self, mode, pInd):
		'''
		Every time loadJson load data for ONE image. The loaded data variable contains fields below:
		# data['info'] = info[1]
		# data['box_scores'] = 256x1
		# data['boxes_pred'] = [~128]x4
		# data['boxes_gt'] = [~128]x4
		# data['box_captions_gt'] = [~128]x15
		# data['box_feats'] = [~128]x512x7x7
		# data['box_codes'] = [~128]x4096
		# data['glob_feat'] = 512x30x45
		# data['glob_caption_gt'] = 100
		'''

		# list data_$pipIndex_*
		# pInd = self.pipeIndex
		data_file = self.dataPipePath+'data_'+str(pInd)+'_*'
		pick_confirm_file = self.dataPipePath+'pick_confirm_'+str(pInd)
		writing_block_file = self.dataPipePath+'writing_block_'+str(pInd)
		reading_block_file = self.dataPipePath+'reading_block_'+str(pInd)

		while True:
			filename = glob.glob(data_file)
			if len(filename)>=1:
				if (not os.path.isfile(pick_confirm_file)) or mode=='test':
					# pick_confirm file has been removed by lua loader. so this file is new. 
					if (not os.path.isfile(writing_block_file)) and (not os.path.isfile(reading_block_file)):
						# lua writing file finished.
						if mode=='train':
							os.mknod(reading_block_file)
						break
		assert(len(filename)==1)
		filename = filename[0]
		# read iter and numiters
		tmp = filename.split('/')[-1].split('_')
		numiters = tmp[-1]
		itr = tmp[-2]

		# json load data_$pipIndex_*
		with open(filename,'rb') as f:
			reader = torchfile.T7Reader(f, utf8_decode_strings=True)
			data = reader.read_obj()

		if mode=='train':
			os.remove(filename)
			os.remove(reading_block_file)
			# place pick_confirm_$pipIndex to notify lua program
			os.mknod(pick_confirm_file)
			

		# update pInd
		# self.updatePipeIndex()
		# return data, iter, numiters
		return data, itr, numiters

	def getBatch(self):
		raise NotImplementedError

	def __len__(self):
		"""Make sure len(dataset) return the size of dataset. Required to override."""
		raise NotImplementedError

	def __getitem__(self, idx):
		"""Support indexing such that dataset[i] get ith sample. Required to override"""
		raise NotImplementedError


class LoaderEnc(_BaseDataLoader):
	"""docstring for LoaderEncTrain"""
	def __init__(self,mode='train'):
		super(LoaderEnc, self).__init__()
		if mode=='train':
			self.init_pick_confirm_files()
		self.mode = mode
		_,_,numiters = self.getBatch(self.pipeLen-1)
		self.numiters = int(numiters)

	def filtReplicate(self, data):
		# shuffle 128 gt boxes copy
		capsGt = data['box_captions_gt']
		# get one box per real gt box
		numCaps = len(capsGt) 
		randInds = list(range(numCaps))
		random.shuffle(randInds)
		capList = []
		selectInds = []
		for i in range(numCaps):
			zzz = list(capsGt[randInds[i]])
			if zzz not in capList:
				capList.append(zzz)
				selectInds.append(randInds[i])
		# return selected inds
		return selectInds
		
	def collate_fn(self,batch): #loader,numImgs=8
		box_feats = []
		box_captions = []
		capLens = []
		numImgs = len(batch)
		def getLengths(caps):
			batchSize = len(caps)
			lengths = torch.zeros(batchSize,dtype=torch.int32)
			for i in range(batchSize):
				cap = caps[i]
				nonz = (cap==0).nonzero()
				lengths[i] = nonz[0][0] if len(nonz)>0 else len(cap)
			return lengths
		for i in range(numImgs):
			data = batch[i][0]
			box_feats.append(torch.tensor(data['box_feats']))
			box_captions.append(torch.LongTensor(data['box_captions_gt']))
			capLens.append(getLengths(box_captions[-1]))
		box_feats = torch.cat(box_feats)
		box_captions = torch.cat(box_captions)
		capLens = torch.cat(capLens)

		# sort decreasing order
		inds = torch.argsort(-capLens)
		box_feats = box_feats[inds]
		box_captions = box_captions[inds]
		capLens = capLens[inds]
		nonz = (capLens<=0).nonzero()
		if len(nonz)>0:
			ending = nonz[0][0]
			box_feats = box_feats[:ending]
			box_captions = box_captions[:ending]
			capLens = capLens[:ending]

		return box_feats, box_captions, capLens

	def getBatch(self, pipIndx):
		'''
		For LSTM encoder training, the "batch" here is actually 
		a batch of boxes and captions for ONE image. 
		Every image has up to 128 boxes, numbers may vary. 
		'''
		data, itr, numiters = self.loadOneJson(self.mode,pipIndx)
		filtInds = self.filtReplicate(data)
		# return only useful data fields
		ret = {}
		ret['info'] = data['info']
		ret['box_feats'] = data['box_feats'][filtInds]
		ret['box_captions_gt'] = data['box_captions_gt'][filtInds]
		ret['glob_feat'] = data['glob_feat']
		if 'glob_caption_gt' in data:
			ret['glob_caption_gt'] = data['glob_caption_gt']

		return ret, itr, numiters
	def __len__(self):
		"""Make sure len(dataset) return the size of dataset. Required to override."""
		return self.numiters

	def __getitem__(self, idx):
		"""Support indexing such that dataset[i] get ith sample. Required to override"""
		return self.getBatch(idx%self.pipeLen)


class LoaderDec(_BaseDataLoader):
	"""docstring for LoaderEncTrain"""
	def __init__(self, mode='train'):
		super(LoaderDec, self).__init__()
		if mode=='train':
			self.init_pick_confirm_files()
		self.mode = mode
		# _,_,numiters = self.getBatch(self.pipeLen-1)
		_,_,numiters = self.getBatch(self.pipeLen-1)
		self.numiters = int(numiters)

	def overlap_area(self,A, B):  # X,Y ,W, H

		W_cen_dis = np.abs(A[0] - B[0])
		H__cen_dis = np.abs(A[1] - B[1])
		W_dis = np.abs(A[2] + B[2]) / 2
		H_dis = np.abs(A[3] + B[3]) / 2

		if ((W_cen_dis < W_dis) or (H__cen_dis < H_dis)):

			overlap_width = np.abs(W_dis - W_cen_dis)
			overlap_height = np.abs(H_dis - H__cen_dis)
			overlapp_area = overlap_height * overlap_width
			area_A = A[2] * A[3]
			area_B = B[2] * B[3]
			ratioA = overlapp_area / area_A
			ratioB = overlapp_area / area_B

			if (ratioA > 0.6 or ratioB > 0.6):
				return False

			else:
				return True
		else:
			return True

	def indep_box_feats(self,sel_ind,num,box_gt):
		for i in sel_ind:
			if self.overlap_area(box_gt[i], box_gt[num]):
				return sel_ind
		sel_ind.append(num)
		return sel_ind

	def randomSample(self, box_gt):
		# diversly distributed sample

		box_feats_seq = list(range(0,len(box_gt)))
		random.shuffle(box_feats_seq)
		# box_feats_list = [box_feats[box_feats_seq[0]]]
		sel_ind = [box_feats_seq[0]]

		for i in box_feats_seq[1:]:
			sel_ind= self.indep_box_feats(sel_ind, i, box_gt)

		return sel_ind

	# def randomSample(self,box_feats,boxes_gt):  # box_feats [3,512,7,7]
	# 	# diversly distributed sample

	# 	# temp = data['box_scores'][0:data['box_feats'].shape[0]] # get the first ~ 128  boxes scores
	# 	# temp = temp[filtInds]
	# 	# temp = temp.reshape(temp.shape[0]*temp.shape[1])
	# 	# Maxindex = np.argsort(temp)[-samplenum:][::-1]

	# 	# sampledData = data['box_feats'][filtInds][Maxindex]
	# 	# sampledData = sampledData[np.newaxis,:]
	# 	# sampledData[0,:,:,:]=pipIndx
		
	# 	scores = np.squeeze(scores)

	# 	if len(scores)<5:
	# 		M,D,H,W = box_feats.shape
	# 		tmp = np.zeros([5,D,H,W])
	# 		tmp[:M] = box_feats
	# 		tmp[M:] = np.tile(box_feats[0],(5-M,1))
	# 		box_feats = tmp
	# 		tmp = np.zeros(5)
	# 		tmp[:M] = scores
	# 		tmp[M:] = scores[0]
	# 		scores = tmp

	# 	scores = scores[:len(box_feats)]
	# 	prob = softmax(scores)
	# 	index = np.random.choice(len(box_feats),5, replace=False,p=prob)

	# 	box_feats = box_feats[index]
		
	# 	return scores,box_feats

	def filtReplicate(self, data):

		capsGt = data['box_captions_gt']
		# get one box per real gt box

		numCaps = len(capsGt)
		randInds = list(range(numCaps))
		random.shuffle(randInds)
		capList = []
		selectInds = []
		for i in range(numCaps):
			zzz = list(capsGt[randInds[i]])
			if zzz not in capList:
				capList.append(zzz)
				selectInds.append(randInds[i])

		return selectInds

	def getBatch(self,pipIndx):  # only this one needs to form a batch
		data, itr, numiters = self.loadOneJson(self.mode, pipIndx)

		filtInds = self.filtReplicate(data)
		# return only useful data fields
		ret = {}
		ret['info'] = data['info']
		ret['box_scores'] = data['box_scores'][filtInds]
		ret['box_feats'] = data['box_feats'][filtInds]
		ret['boxes_gt'] = data['boxes_gt'][filtInds]
		ret['glob_feat'] = data['glob_feat']

		sampledInds = self.randomSample(ret['boxes_gt'])
		ret['box_scores'] = ret['box_scores'][sampledInds]
		ret['box_feats'] = ret['box_feats'][sampledInds]
		ret['boxes_gt'] = ret['boxes_gt'][sampledInds]

		return ret, itr, numiters


	def collate_fn(self,batch): #loader,numImgs=8
		numBoxes = int(np.random.choice(4, 1)+2)
		box_feats = []
		box_global_feats=[]
		numImgs = len(batch)
		for i in range(numImgs):
			data = batch[i][0]
			numSps,D,H,W = data['box_feats'].shape
			box_feats.append(torch.tensor(data['box_feats'][:(numSps-numSps%numBoxes)].view(-1,numBoxes,D,H,W)))
			box_global_feats += [torch.tensor(data['glob_feat'])]*len(box_feats[-1])
		box_feats = torch.cat(box_feats,dim=0)

		# box_global_feats = torch.cat(box_global_feats)
		# _,B,C = box_global_feats.shape
		# box_global_feats=box_global_feats.reshape(numImgs,A,B,C )

		return box_feats, box_global_feats, numBoxes


	def __len__(self):
		"""Make sure len(dataset) return the size of dataset. Required to override."""
		return self.numiters

	def __getitem__(self, idx):
		"""Support indexing such that dataset[i] get ith sample. Required to override"""
		return self.getBatch(idx%self.pipeLen)




if __name__ == '__main__':
	loader = LoaderDec()
	cnt = 0
	while True:
		data, itr, numiters = loader.getBatch()
		print('Iter'+str(itr)+'Get data for image: '+str(data['info']['filename']))

