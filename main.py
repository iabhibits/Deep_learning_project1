import argparse
import glob
import logging
import os
import random
import sys

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
from train import Model

def read_test_data(filename):
	data = pd.read_csv(filename,sep="\n", header=None)
	#print(data.shape)
	train_x = []
	for i in tqdm(range(data.shape[0])):
		j = int(data[0][i])
		binary_rep = [j >> d & 1 for d in range(10)]
		#print("j {} and bin {} \n ".format(j,binary_rep))
		train_x.append(binary_rep)
	train_x = np.array(train_x)
	input_feature = torch.tensor(train_x,dtype=torch.float)
	#print(input_feature)
	w = torch.Tensor([0.1,0.2,0.3,0.4])
	dataset = TensorDataset(input_feature)
<<<<<<< HEAD
	return dataset,w,data

def evaluate(model,eval_dataset,device,w,data):
=======
	return dataset,w

def evaluate(model,eval_dataset,device,w):
>>>>>>> 06fdb4eeaba0d8338b6f2aa807e38d6b667b172a
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
	#print(preds.shape)
	for i in range(preds.shape[0]):
		r = np.argmax(preds[i])
		#print(r)
		if r == 0:
<<<<<<< HEAD
			print(int(data[0][i]))
			eval_result.append(int(data[0][i]))
=======
			print(i+1)
			eval_result.append(i+1)
>>>>>>> 06fdb4eeaba0d8338b6f2aa807e38d6b667b172a
		elif r == 1:
			print('fizz')
			eval_result.append('fizz')
		elif r == 2 :
			print('buzz')
			eval_result.append('buzz')
		elif r == 3:
			print("fizzbuzz")
			eval_result.append("fizzbuzz")
	eval_result = np.array(eval_result)
	# dataset = pd.DataFrame({eval_result})
	# dataset.to_csv('test_output.txt', sep='\t', index=False)
	repo_path = os.path.dirname(os.path.abspath(__file__))
	out_sw_2_fp = repo_path + "/Software2.txt"
	np.savetxt(out_sw_2_fp, eval_result, fmt="%s")
	return 0

def main():
	print("Abhishek Kumar")
	print("15648")
	print("CSA")

	parser = argparse.ArgumentParser()
	parser.add_argument("--test-data", default='test_input.txt', type=str, help="Name of the test file")
	args = parser.parse_args()
	
	model_path='model/model.bin'
	model = Model(h=4)
	device = "cpu"
	model.to(device)
<<<<<<< HEAD
	eval_dataset,w,data = read_test_data(args.test_data)
	model.load_state_dict(torch.load(model_path))
	evaluate(model, eval_dataset,device,w,data)
=======
	eval_dataset,w = read_test_data(args.test_data)
	model.load_state_dict(torch.load(model_path))
	evaluate(model, eval_dataset,device,w)
>>>>>>> 06fdb4eeaba0d8338b6f2aa807e38d6b667b172a

if __name__ == '__main__':
	main()