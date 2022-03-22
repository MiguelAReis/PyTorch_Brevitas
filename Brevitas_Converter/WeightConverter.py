import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os


parser = argparse.ArgumentParser(description='Weights Extractor')
parser.add_argument('--original', default="original.pth", type=str, help='Original PTH file with trained weights')
parser.add_argument('--blank', default="blank.pth", type=str, help='Blank Quantized PTH file')
parser.add_argument('--out', default="out.pth", type=str, help='Output file with quantized weights')
args = parser.parse_args()

original = torch.load(args.original)
blank = torch.load(args.blank)

for key, value in original["net"].items():
	for key_, value_ in blank["net"].items():
		if(key==key_):
			with torch.no_grad():
				value_.data=value.clone().detach()
			print("FOUND")
			print(key+str(value_))
			print(key_+str(value))
			break;

torch.save(blank,args.out)