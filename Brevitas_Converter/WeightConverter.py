import argparse
import os
import numpy as np
from tqdm import tqdm

import brevitas.nn as qnn
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from TrainedModel import lenet
from TrainedModel import lenetquant

model=torch.nn.DataParallel(lenet.LeNet())
modelQuant=torch.nn.DataParallel(lenetquant.LeNetQuant())

state_dict = torch.load("TrainedModel/lenet67%teste.pth", map_location='cpu')

model.load_state_dict(state_dict['net'],strict=True)
model.eval()

stateQuant = {
    'net': modelQuant.state_dict()
}
torch.save(stateQuant, './TrainedModel/blankQuant.pth')
modelQuant

#torch.set_printoptions(profile="full")


for name, param in model.named_parameters():
	print("Original " +name)
	for nameQ, child in modelQuant.named_children():
		for nameQ_, param_ in child.named_parameters():
			if(nameQ_ in name):
				print("New "+ nameQ_)
				print("Before")
				print(param_)
				with torch.no_grad():
					param_.data=param
				print("After")
				print(param_)

for name, param in model.named_parameters():
	print("Original " +name)
	for nameQ, child in modelQuant.named_children():
		for nameQ_, param_ in child.named_parameters():
			if(nameQ_ in name):
				print("New "+ nameQ_)
				print("test")
				print(param_)
			#if("quant_inp.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value" == nameQ_):
				#print("TRUE")
				#print(nameQ_)
				#param_= 
				#print(param_)






stateQuant["net"]["quant_inp.act_quant.fused_activation_quant_proxy.tensor_quant.scaling_impl.value"]=torch.tensor(2.73, requires_grad=True)


from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in stateQuant["net"].items():
    name = k.replace("module.","",1) # remove `module.`
    new_state_dict[name] = v
# load params

for k, v in new_state_dict.items():
	print(k)
ckpt = OrderedDict()
ckpt["net"]=new_state_dict
torch.save(ckpt, './TrainedModel/newQuant.pth')