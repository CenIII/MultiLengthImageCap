import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LMDataset(Dataset):
	"""docstring for LMDataLoader"""
	def __init__(self, vocab_data, fullcaps):
		super(LMDataset, self).__init__()
		self.data = fullcaps
		self.wordDict = vocab_data['word_dict']
		self.embedding = vocab_data['word_embs']
		self.sos_id = self.wordDict['<START>']
		self.eos_id = self.wordDict['<END>']
		self.pad_id = self.wordDict['<PAD>']

	def seq_collate(self, batch):
		"""Pack a series of samples into a batch. Each Sample is a tuple (brkSentence, [style], sentence, marker).

		The default collate_fn of Pytorch use torch.stack() assuming every sample has the same size.
		For this task, the length of sentences may vary, so do the sample generated.
		See https://jdhao.github.io/2017/10/23/pytorch-load-data-and-make-batch/ for more information.

		Returns:
			A dict of list with all its member being a list of length batch_size
		"""
		# print('>>>>>>>batch: '+str(batch))
		batchSize = len(batch)
		
		maxLen = 0
		lengths = []
		for seq in batch:
			seqLen = len(seq)
			lengths.append(seqLen)
			if seqLen > maxLen:
				maxLen = seqLen
		packed = np.zeros([batchSize, maxLen])
		mask = np.zeros([batchSize, maxLen])
		for i in range(batchSize):
			packed[i][:lengths[i]] = batch[i]
		# lengths = np.array(lengths)
		# inds = np.argsort(lengths)[::-1]
		sent = torch.LongTensor(packed)
		# mask = torch.tensor(mask)

		return {'sentence':sent}

	def word2indices(self, sList, sos=False):
		"""For each sentence in a list of sentences, convert its tokens into word index including auxilary tokens.
		Args:
			sList: A list of sentences.
			sos: A boolean indicating whether to add <sos> token at the start of sentences.
		"""
		resList = []
		for sentence in sList:
			indArr = []
			if sos:
				indArr.append(self.sos_id)
			for i in range(len(sentence)):
				word = sentence[i]
				if word in self.wordDict:
					indArr.append(self.wordDict[word])
			indArr.append(self.eos_id) 
			indArr = np.array(indArr)
			resList.append(indArr)
		return resList

	def __len__(self):
		"""Make sure len(dataset) return the size of dataset. Required to override."""
		return len(self.data)

	def __getitem__(self, idx):
		"""Support indexing such that dataset[i] get ith sample. Required to override"""
		sentence = self.data[idx]
		return self.word2indices([sentence])[0]

	def getLoader(self, batchSize=128, shuffle=True):
		return DataLoader(self,batch_size=batchSize, shuffle=shuffle, num_workers=2, collate_fn=self.seq_collate)