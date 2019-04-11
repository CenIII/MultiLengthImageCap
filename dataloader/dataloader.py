# a python dataloader that loads dumped features files prodeced by lua program.
import os
import glob
# import json
import torchfile
import random
from torch.utils.data import Dataset
import torch

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class _BaseDataLoader(Dataset):
	"""docstring for BaseDataLoader"""
	def __init__(self):
		super(_BaseDataLoader, self).__init__()
		# self.pipeIndex = 0
		# save pick_confirm for all pip indices
		self.dataPipePath = './densecap/data_pipeline/'
		self.pipeLen = 30

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
				lengths[i] = (cap==0).nonzero()[0][0]
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


		
class _LoaderDec(_BaseDataLoader):
	"""docstring for LoaderDec"""
	def __init__(self, arg):
		super(_LoaderDec, self).__init__()
		self.init_pick_confirm_files()
		self.arg = arg

	def randomSample(self,data,boxes,sampleNum):
		# diversly distributed sample 

		return sampledData
		
class LoaderDecTrain(_LoaderDec):
	"""docstring for LoaderEncTrain"""
	def __init__(self, arg):
		super(LoaderDecTrain, self).__init__()
		self.arg = arg
	def filtReplicate(self, data):
		# shuffle 128 gt boxes copy

		# get one box per real gt box

		# return selected inds
		pass
	def getBatch(self):  # only this one needs to form a batch
		pass


class LoaderDecTest(_LoaderDec):
	"""docstring for LoaderEncTrain"""
	def __init__(self, arg):
		super(LoaderDecTest, self).__init__()
		self.arg = arg
	def getBatch(self):
		pass

if __name__ == '__main__':
	loader = LoaderEncTrain()
	cnt = 0
	while True:
		data, itr, numiters = loader.getBatch()
		print('Iter'+str(itr)+'Get data for image: '+str(data['info']['filename']))

