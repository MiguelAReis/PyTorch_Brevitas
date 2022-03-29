import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = image.clone().detach().requires_grad_(True)
    image = image.unsqueeze(0)
    return image



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--input', default="testImages/input.png", type=str, help='Input Image')
parser.add_argument('--ckpt', default="checkpoint/ckpt.pth", type=str, help='input net pth file')
parser.add_argument('--weights', default=8, type=int, help='batch size')
parser.add_argument('--activations', default=8, type=int, help='batch size')
args = parser.parse_args()
 
classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transformNotNormal = transforms.Compose([
    transforms.ToTensor(),
])

device ="cuda"
print('==> Building model..')
print("Weights = "+ str(args.weights) +", Activations = "+str(args.activations))
MobileNetQuant.setBitWidths(args.weights,args.activations)
net = MobileNetQuant()
net = net.to(device)


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load checkpoint.
print('==> Loading checkpoint..')
checkpoint = torch.load(args.ckpt)
try:
    net.module.load_state_dict(checkpoint["net"])
except:
    net.load_state_dict(checkpoint["net"])

    
net.to(device)

net.eval()
if args.input != "RANDOM":
	evalClass=np.argmax(net(image_loader(transform, args.input)).cpu().detach().numpy())


	print("Class is : "+classes[evalClass])
	image = Image.open(args.input)
	plt.imshow(image)
	plt.show()
