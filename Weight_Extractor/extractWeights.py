import argparse
import os
import numpy as np
from tqdm import tqdm

import brevitas.nn as qnn
import math

parser = argparse.ArgumentParser(description='Weights Extractor')
parser.add_argument('--pth', default="./Example/example.pth", type=str, help='pth location')
parser.add_argument('--weights', default=8, type=int, help='weights bit width')
parser.add_argument('--activations', default=8, type=int, help='activations bit width')
parser.add_argument('--folder', default="Example", type=str, help='activations bit width')
parser.add_argument('--model', default="ResNet101Quant", type=str, help='activations bit width')
args = parser.parse_args()

exec("from "+ args.folder + " import *")
model = eval(args.model+"(weights=args.weights,activations=args.activations)")
#model = ResNet101Quant(weights=args.weights,activations=args.activations)



state_dict = torch.load(args.pth, map_location='cpu')

from collections import OrderedDict
ckpt = OrderedDict()
for k, v in state_dict["net"].items():
    name = k.replace("module.","",1) # remove `module.`
    ckpt[name] = v

model.load_state_dict(ckpt,strict=True)
model.eval()


#model.eval()
torch.set_printoptions(profile="full")

for name, child in model.named_children():
    for name_, param in child.named_parameters():
        if "conv" in name or "conv" in name_:
            if name_!="weight":
                layerName="model."+name+'['+ name_.replace('.','].',1).replace("weight","int_weight()",1)
            else:
                layerName="model."+name+".int_weight()"
            print(layerName)
            tensors= eval(layerName)
            scale=eval(layerName.replace("int_weight()","quant_weight_scale()",1))
            scale=scale.detach().numpy()
            print(tensors*scale)
            print("Scale is "+str(scale)+ "\nFixed Point radix is "+ str(math.log(scale,2)) +" positions to the right")
            input("Press Enter to continue...\n\n\n\n")
