import argparse
import glob
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

from train import generate_training_data


class Model(nn.Module):
	def __init__(self,h=None):
		super(Model, self).__init__()
		self.linearlayer = nn.Linear(10,100)
		#self.linear = nn.Linear(100,10)
		self.classifier = nn.Linear(100,h)

	def forward(self,feature=None, labels=None,wt = None):
		# feature = feature.dtype(torch.int64)
		#feature = feature.type(torch.LongTensor)
		x = nn.functional.relu(self.linearlayer(feature))
		#x = nn.functional.relu(self.linear(x))
		logits = self.classifier(x)

		reshaped_logits =  logits.view(-1,4)
		outputs = (logits,)

		if labels is None:
			loss = 0
			outputs = (loss, ) + outputs

		if labels is not None:
			loss_fct = CrossEntropyLoss(wt)
			loss = loss_fct(reshaped_logits, labels)
			outputs = (loss, ) + outputs
		return outputs

def evaluate(model,eval_dataset,device,w):
	results = {}
	preds = None
	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size = 10)
	for batch in tqdm(eval_dataloader, desc="Evaluating"):

		model.eval()
		batch = tuple(t.to(device) for t in batch)

		with torch.no_grad():
			inputs = { 
						"feature":batch[0].unsqueeze(dim=1),
						"wt" : w,
			        }
			outputs = model(**inputs)
			#print(outputs)
			tmp_eval_loss, logits = outputs[:2]

			if preds is None:
				preds = logits.detach().cpu().numpy()
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
	eval_result = []
	print(preds.shape)
	for i in range(preds.shape[0]):
		r = np.argmax(preds[i])
		#print(r)
		if r == 0:
			print(i+1)
			eval_result.append([i+1,i])
		elif r == 1:
			print('fizz')
			eval_result.append([i+1,'fizz'])
		elif r == 2 :
			print('buzz')
			eval_result.append([i+1,'buzz'])
		elif r == 3:
			print("fizzbuzz")
			eval_result.append([i+1,"fizzbuzz"])
	eval_result = np.array(eval_result)
	dataset = pd.DataFrame({'feature' : eval_result[:,0],'label': eval_result[:,1]})
	dataset.to_csv('output.csv', sep='\t')
	return 0

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--test-data", default='test.txt', type=str, help="Name of the test file")
	args = parser.parse_args()
	model_path='model/model.bin'
	model = Model(h=4)
	device = "cpu"
	model.to(device)
	eval_dataset,w = generate_training_data(1,101)
	model.load_state_dict(torch.load(model_path))
	evaluate(model, eval_dataset,device,w)

if __name__ == '__main__':
	main()