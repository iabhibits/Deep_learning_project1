import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import argparse
import os


def gen_data():
	x = np.arange(1,100)
	return np.random.permutation(x)

def main():
	eval_result = []
	data = gen_data()
	for i in tqdm(data):
		if i%15 == 0:
			# print("fizzbuzz")
			eval_result.append('fizzbuzz')
		elif i%3 == 0 :
			# print("fizz")
			eval_result.append('fizz')
		elif i%5 == 0:
			# print("buzz")
			eval_result.append("buzz")
		else:
			# print(i)
			eval_result.append(i)

	eval_result = np.array(eval_result)
	repo_path = os.path.dirname(os.path.abspath(__file__))
	out_sw_1_fp = repo_path + "/target.txt"
	np.savetxt(out_sw_1_fp, eval_result, fmt="%s")
	np.savetxt("input.txt",data,fmt="%s")


if __name__ == '__main__':
	main()


