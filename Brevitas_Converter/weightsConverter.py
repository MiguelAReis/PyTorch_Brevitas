import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os


parser = argparse.ArgumentParser(description='Converts weights from a trained model into a quantized model generated by brevitasConverter')
parser.add_argument('--original', default="original.pth", type=str, help='Original PTH file with trained weights')
parser.add_argument('--blank', default="blank.pth", type=str, help='Blank quantized PTH file')
parser.add_argument('--out', default="out.pth", type=str, help='Output file with quantized weights')
args = parser.parse_args()

original = torch.load(args.original)
blank = torch.load(args.blank)
totalLayers=0
transferedLayers=0
totalBlank=0

state_dict=""
for key, value in original.items():
	if(key in ("state_dict","statedict","net","network")):
		state_dict=key
	else:
		blank[key]=value
if(state_dict==""):
	raise RuntimeError("State_dict String couldnt be found")

if(state_dict != "net"):
	blank[state_dict]=blank.pop("net")

originalModule=False
if("module" in list(original[state_dict].items())[0][0]):
	originalModule=True
for key, value in original[state_dict].items():
	totalLayers=totalLayers+1
	for key_, value_ in blank[state_dict].items():
		if(key.replace("module.","",1) == key_):
			with torch.no_grad():
				value_.data=value.clone().detach()
				transferedLayers=transferedLayers+1
				copied=True
			break


blankCopy=blank[state_dict].copy()

for key, value in blankCopy.items():
	if(originalModule):
		newKey="module."+key
		blank[state_dict][newKey]=blank[state_dict][key]
		del blank[state_dict][key]
	totalBlank=totalBlank+1


torch.save(blank,args.out)
print("Weights and Bias Converted to "+args.out)
print("A total of "+str(transferedLayers)+" layers were transfered out of "+str(totalLayers))
print("A total of "+str(totalBlank-transferedLayers)+ " layers were left blank")

