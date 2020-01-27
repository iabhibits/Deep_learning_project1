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

def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

def generate_training_data(l=101,h=1001):
	"Create a training dataset for fizzbuzz problem"
	c0,c1,c2,c3 = 0,0,0,0
	train_x = []
	train_labels = []
	train_x.append([0,0,0,0,0,0,0,0,0,0])
	train_labels.append(0)
	for i in tqdm(range(l,h)):
	    binary_rep = [i >> d & 1 for d in range(10)]
	    train_x.append(binary_rep)
	    if i%15 == 0:
	        #x = [0,0,0,1]
	        x = 3
	        c3 += 1
	    elif i%3 == 0 :
	        #x = [0,1,0,0]
	        x = 1
	        c1 += 1 
	    elif i%5 == 0:
	        #x = [0,0,1,0]
	        x = 2
	        c2 += 1
	    else:
	        #x = [1,0,0,0]
	        x = 0
	        c0 += 1

	    train_labels.append(x)
	train_x = np.array(train_x)
	train_labels = np.array(train_labels)
	input_feature = torch.tensor(train_x,dtype=torch.float)
	input_labels = torch.tensor(train_labels, dtype=torch.long)
	dataset = TensorDataset(input_feature,input_labels)
	s = c0 + c1 + c2 + c3
	w1 = s/c1
	w2 = s/c2
	w3 = s/c3
	w0 = s/c0
	#w = torch.Tensor(w0,w1,w2,w3)
	w = torch.Tensor([0.1,0.2,0.3,0.4])
	return dataset,w

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

def simple_accuracy(preds, labels):
	#preds = np.argmax(preds, axis = 0)
	r = []
	for i in range(preds.shape[0]):
		r.append(np.argmax(preds[i]))
	return (r == labels).mean()

def train(args, train_dataset, model,w):
	tb_writer = SummaryWriter()

	train_sampler = SequentialSampler(train_dataset)
	train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size = args.train_batch_size)
	t_total = len(train_dataloader) // args.num_train_epochs


	#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	optimizer = optim.Adam(model.parameters(), lr=0.01, betas=[0.9,0.98])

	# Train!
	logger.info("***** Running training *****")
	logger.info("  Num examples = %d", len(train_dataset))
	logger.info("  Num Epochs = %d", args.num_train_epochs)
	logger.info("  Total optimization steps = %d", t_total)

	global_step = 0
	tr_loss, logging_loss = 0.0, 0.0
	best_dev_acc = 0.0
	best_steps = 0
	model.zero_grad()

	train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
	set_seed(args)

	for _ in train_iterator:
		epoch_iterator = tqdm(train_dataloader, desc="Iteration")
		for step, batch in enumerate(epoch_iterator):
			model.train()
			batch = tuple(t.to(args.device) for t in batch)
			inputs = { 
						"feature":batch[0].unsqueeze(dim=1),
						"labels": batch[1],
						"wt": w,
			}

			outputs = model(**inputs)
			loss = outputs[0]

			loss.backward()
			optimizer.step()
			tr_loss += loss.item()

			model.zero_grad()
			global_step += 1
		print(loss.item())

	tb_writer.close()

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	output_dir = os.path.join(args.output_dir, "model.bin")
	torch.save(model.state_dict(), output_dir)
	return global_step, tr_loss / global_step, best_steps

def evaluate(model,eval_dataset,device,w):
	results = {}
	preds = None
	eval_sampler = SequentialSampler(eval_dataset)
	eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size = 10)
	eval_loss = 0
	nb_eval_steps = 0
	for batch in tqdm(eval_dataloader, desc="Evaluating"):

		model.eval()
		batch = tuple(t.to(device) for t in batch)

		with torch.no_grad():
			inputs = { 
						"feature":batch[0].unsqueeze(dim=1),
						"labels": batch[1],
						"wt" : w,
			        }
			outputs = model(**inputs)
			#print(outputs)
			tmp_eval_loss, logits = outputs[:2]

			eval_loss += tmp_eval_loss.mean().item()
			nb_eval_steps += 1
			if preds is None:
				preds = logits.detach().cpu().numpy()
				out_label_ids = inputs["labels"].detach().cpu().numpy()
			else:
				preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
				out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
	eval_loss = eval_loss / nb_eval_steps

	acc = simple_accuracy(preds, out_label_ids)
	# print("Loss : {} \n".format(eval_loss))
	# print("accuracy : {} \n".format(acc))
	result = {"eval_acc": acc, "eval_loss": eval_loss}
	print(result)
	results.update(result)
	eval_result = []
	print(preds.shape)
	for i in range(1,preds.shape[0]):
		r = np.argmax(preds[i])
		#print(r)
		if r == 0:
			print(i)
			eval_result.append(i)
		elif r == 1:
			print('fizz')
			eval_result.append('fizz')
		elif r == 2 :
			print('buzz')
			eval_result.append('buzz')
		elif r == 3:
			print("fizzbuzz")
			eval_result.append('fizzbuzz')
	eval_result = np.array(eval_result)
	dataset = pd.DataFrame({'label': eval_result})
	dataset.to_csv('output.csv', sep='\t')
	return eval_loss

def main():
	parser = argparse.ArgumentParser()

	# Required Parameters

	parser.add_argument("--train_batch_size", default=32, type=int, help="What is the training batch size?")
	parser.add_argument("--num_train_epochs", default=32, type=int, help="Total number of training epochs to perform.")
	parser.add_argument("--logging_steps", default=500, type=int, help="Log every X update steps")
	parser.add_argument("--save_steps", default=500, type=int, help="Save checkpoints every X steps")
	parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
	parser.add_argument("--output_dir", default = 'save/', type=str, help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
	parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")


	args = parser.parse_args()

	device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args.device = device
	print("The cuda device being used is : {} \n".format(args.device))
	set_seed(args)

	if args.do_train:

		train_dataset,w = generate_training_data(101,1001)
		print(w)
		model = Model(h=4)
		model.to(args.device)
		global_step, tr_loss, best_steps = train( args, train_dataset, model,w)
		logger.info("global_steps = %s, average_loss = %s ", global_step, tr_loss)
		eval_dataset,w = generate_training_data(1,101)
		evaluate(model, eval_dataset,device,w)


if __name__ == "__main__":
	main()










