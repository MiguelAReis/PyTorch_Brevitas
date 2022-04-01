import argparse
import os
import numpy as np
from tqdm import tqdm

import brevitas.nn as qnn
import math
parser = argparse.ArgumentParser(description='Creates a blank pth file to be used with weightsConverter.py')
parser.add_argument('--modelFile', default="ResNet101Quant", type=str, help='Input model description file location e.g. mobilenetQuant.py ')
parser.add_argument('--model', default="ResNet101Quant", type=str, help='Name of the model e.g. MobileNetQuant')
parser.add_argument('--output', default="./out.pth", type=str, help='Output file e.g. blank.pth')
parser.add_argument('--modelParams', default="", type=str, help='Parameters of the Model')
args = parser.parse_args()

exec("from "+args.modelFile.replace(".py","",1)+" import *")
eval(args.model+".setBitWidths(8,8)")
model = eval(args.model+"("+args.modelParams+")")


state = {
    'net': model.state_dict(),
    'optimizer_state': None,
    'acc': 0,
    'epoch': 0,
}

torch.save(state, args.output)
print(args.output+" file created")
