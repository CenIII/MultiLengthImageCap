# a python dataloader that loads dumped features files prodeced by lua program.
import os
import glob
# import json
import torchfile

class _BaseDataLoader(object):
	"""docstring for BaseDataLoader"""
	def __init__(self):
		super(_BaseDataLoader, self).__init__()
		self.pipeIndex = 0
		# save pick_confirm for all pip indices
		self.dataPipePath = '../densecap/data_pipeline/'
		self.pipeLen = 10
		self.init_pick_confirm_files()

	def updatePipeIndex(self):
		self.pipeIndex = (self.pipeIndex+1)%self.pipeLen

	def init_pick_confirm_files(self):
		for i in range(self.pipeLen):
			save_file = os.path.join(self.dataPipePath,'pick_confirm_'+str(i))
			with open(save_file,'w') as f:
				f.write('a')

	def loadOneJson(self):
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
		pInd = self.pipeIndex
		while True:
			filename = glob.glob(self.dataPipePath+'data_'+str(pInd)+'*')
			if len(filename)>=1:
				if not os.path.isfile(self.dataPipePath+'pick_confirm_'+str(pInd)):
					# pick_confirm file has been removed by lua loader. so this file is new. 
					if not os.path.isfile(self.dataPipePath+'writing_block_'+str(pInd)):
						# lua writing file finished.
						break
		assert(len(filename)==1)
		filename = filename[0]
		print(filename)
		# read iter and numiters
		tmp = filename.split('/')[-1].split('_')
		numiters = tmp[-1]
		itr = tmp[-2]

		# json load data_$pipIndex_*
		with open(filename,'rb') as f:
			reader = torchfile.T7Reader(f, utf8_decode_strings=True)
			data = reader.read_obj()

		os.remove(filename)
		# place pick_confirm_$pipIndex to notify lua program
		with open(self.dataPipePath+'pick_confirm_'+str(pInd), 'w') as f:
			f.write('a')

		# update pInd
		self.updatePipeIndex()
		# return data, iter, numiters
		return data, itr, numiters

	def getBatch(self):
		raise NotImplementedError


class LoaderEncTrain(_BaseDataLoader):
	"""docstring for LoaderEncTrain"""
	def __init__(self):
		super(LoaderEncTrain, self).__init__()
		
	def getBatch(self):
		'''
		For LSTM encoder training, the "batch" here is actually 
		a batch of boxes and captions for ONE image. 
		Every image has up to 128 boxes, numbers may vary. 
		'''
		data, itr, numiters = self.loadOneJson()
		# return only useful data fields
		ret = {}
		ret['info'] = data['info']
		ret['box_feats'] = data['box_feats'] 
		ret['box_captions_gt'] = data['box_captions_gt']
		ret['glob_feat'] = data['glob_feat']
		if 'glob_caption_gt' in data:
			ret['glob_caption_gt'] = data['glob_caption_gt']

		return ret, itr, numiters


		
class _LoaderDec(_BaseDataLoader):
	"""docstring for LoaderDec"""
	def __init__(self, arg):
		super(LoaderDec, self).__init__()
		self.arg = arg

	def randomSample(self,data,boxes,sampleNum):
		# diversly distributed sample 

		return sampledData
		
class LoaderDecTrain(_LoaderDec):
	"""docstring for LoaderEncTrain"""
	def __init__(self, arg):
		super(LoaderEncTrain, self).__init__()
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
		super(LoaderEncTrain, self).__init__()
		self.arg = arg
	def getBatch(self):
		pass

if __name__ == '__main__':
	loader = LoaderEncTrain()
	cnt = 0
	while True:
		data, itr, numiters = loader.getBatch()
		print('Iter'+str(itr)+'Get data for image: '+str(data['info']['filename']))

