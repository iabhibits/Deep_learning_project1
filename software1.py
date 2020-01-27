import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
import argparse


def main():
	for i in tqdm(range(1,100)):
		if i%15 == 0:
			print("fizzbuzz")
		elif i%3 == 0 :
			print("fizz")
		elif i%5 == 0:
			print("buzz")
		else:
			print(i)



if __name__ == '__main__':
	main()


