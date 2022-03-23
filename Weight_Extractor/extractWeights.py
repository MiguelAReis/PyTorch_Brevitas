import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

import brevitas.nn as qnn
import math

parser = argparse.ArgumentParser(description='Quantized Weights Extractor')
parser.add_argument('--pth', default="ckpt.pth", type=str, help='Location of .pth file e.g. ckpt.pth')
parser.add_argument('--weights', default=8, type=int, help='Weights bit width')
parser.add_argument('--activations', default=8, type=int, help='Activations bit width')
parser.add_argument('--modelFile', default="lenetQuant", type=str, help='Input Model Description file e.g. lenetQuant.py')
parser.add_argument('--model', default="LeNetQuant", type=str, help='Name of the Model e.g LeNetQuant')
parser.add_argument('--fullprint',action='store_true',help='Print Full tensors')
args = parser.parse_args()

exec("from "+args.modelFile.replace(".py","",1)+" import *")
eval(args.model+".setBitWidths(args.weights,args.activations)")
model = eval(args.model+"()")
model = torch.nn.DataParallel(model)
state_dict = torch.load(args.pth, map_location='cpu')
model.load_state_dict(state_dict["net"])

model.to("cuda")
model.eval()
cache_inference_quant_bias=True
with torch.no_grad():
    inputs = torch.empty(2,3,32,32)
    inputs.to("cuda")
    model(inputs)

if(args.fullprint):
    torch.set_printoptions(profile="full")


for key, value in state_dict["net"].items():
    if("weight" in key):
        print(eval("model."+key.replace("weight","int_weight()",1)))
        print("Layer Name = "+ key)
        print("Scale = " + str(eval("model."+key.replace("weight","quant_weight_scale()",1))))
        input("ENTER")
    '''
    if("bias" in key):
        print("model."+key.replace(".bias","",1)+".int_bias()")
        print(eval("model."+key.replace(".bias","",1)+".int_bias()"))
        print("Layer Name = "+ key)
        #print("Scale = " + str(eval("model."+key.replace("bias","maybe_quant_bias_scale()",1))))
        input("ENTER")
    '''
    if("relu" in key):
        layerstr=key[:key.rfind("act_quant")-1]
        print("Layer Name = "+ layerstr)
        print("Scale = " + str(eval("model."+layerstr+".quant_act_scale()")))
        input("ENTER")