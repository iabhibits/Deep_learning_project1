import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import argparse
import os

def check_accuracy(target, predicted):
	return (target == predicted).sum()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--target-file", default='test_input.txt', type=str, required=True, help="Name of the ground_truth file")
	parser.add_argument("--check-file", default='test_input.txt', type=str,required=True, help="Name of the predicted file")
	args = parser.parse_args()

	target = pd.read_csv(args.target_file,sep="\n",header=None)
	predicted = pd.read_csv(args.check_file,sep="\n",header=None)

	acc = check_accuracy(target[0],predicted[0])

	print("Accuracy of the predicted file is {}%\n".format(acc))

if __name__ == '__main__':
	main()
