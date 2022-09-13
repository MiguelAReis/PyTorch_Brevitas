import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision

import brevitas.nn as qnn
import brevitas
import math
from collections import OrderedDict


def setCache(m):
    m.cache_inference_quant_bias=True




parser = argparse.ArgumentParser(description='Quantized Weights Extractor')
parser.add_argument('--pth', default="ckpt.pth", type=str, help='Location of .pth file e.g. ckpt.pth')
parser.add_argument('--weights', default=8, type=int, help='Weights bit width')
parser.add_argument('--activations', default=8, type=int, help='Activations bit width')
parser.add_argument('--modelFile', default="lenetQuant", type=str, help='Input Model Description file e.g. lenetQuant.py')
parser.add_argument('--model', default="LeNetQuant", type=str, help='Name of the Model e.g LeNetQuant')
parser.add_argument('--fullprint',action='store_true',help='Print Full tensors')
parser.add_argument('--modelParams', default="", type=str, help='Parameters of the Model')
args = parser.parse_args()

exec("from "+args.modelFile.replace(".py","",1)+" import *")
eval(args.model+".setBitWidths(args.weights,args.activations)")
model =  eval(args.model+"("+args.modelParams+")")
#model = torch.nn.DataParallel(model)

model.apply(setCache)



state_dict = torch.load(args.pth, map_location='cpu')

model.load_state_dict(state_dict["state_dict"])

model.to("cpu")
model.eval()

inputs = torch.zeros(1,3,300,300)
inputs.to("cpu")
model(inputs)


if(args.fullprint):
    torch.set_printoptions(profile="full")

descriptionObj = open("description.txt", mode='w')
filecounter=0
memAddress=0


for key, value in state_dict["state_dict"].items():
    #print(key)
    res=""
    for i in key.split("."):
        
        if i.isdigit():
            i="["+i+"]"
            res=res[:len(res)-1]+i+"."
        else:
            res=res+i+"."

    key=res[:len(res)-1]
    if("weight" in key):
        #print(eval("model."+key.replace("weight","int_weight()",1)))
        layerName=key.replace("weight","int_weight()",1)

        array= eval("model."+key.replace("weight","int_weight()",1))
        array = array.cpu().numpy().astype(int)
        #print(array)
        #array = np.transpose(array,(0,2,3,1))
        #print(np.vectorize(hex)(array))
        if("depthwise" in key):
            array = np.transpose(array,(1,2,3,0))
        else:
            array = np.transpose(array,(0,2,3,1))
        #print(array.shape)
        #print(array)
        numFilters=array.shape[0]
        numKernels=array.shape[3]
        numY=array.shape[1]
        numX=array.shape[2]
        print("Layer Name = "+ key)
        #print("filters is " +str(numFilters) +" kernels is " +str(numKernels) +" Y is " +str(numY) +" X is " +str(numX))
        descriptionObj.write("Layer Name = "+ key + "\n")
        scale=eval("model."+key.replace("weight","quant_weight_scale().cpu().detach().numpy()",1))
        descriptionObj.write("Scale = " + str(scale)+ "\n")
        descriptionObj.write("Shift " + str(abs(math.log(scale,2))) + " positions to the right\n")
        descriptionObj.write("Address is " + hex(memAddress)+"\n")
        zvalues= math.floor(((numKernels-1)/math.floor(64/int(args.weights)))+1)
        #print("zvalues is " + str(zvalues))
        remainder = 64%int(args.weights)
        #print("remainder is " + str(remainder))
        #print("max is " + str(np.amax(array)))
        #print("min is " + str(np.amin(array)))
        outarray = np.zeros((numFilters, numY, numX, zvalues),dtype=np.int64)
        memAddress=memAddress+(numFilters*numY*numX*zvalues*8)
        #print(outarray.shape)
        zindex = 0
        values=0

        kiters=math.ceil(numKernels/math.floor(64/int(args.weights)))*math.floor(64/int(args.weights))
        #print("kiters is " + str(kiters))
        for f in range(numFilters):
            for y in range(numY):
                for x in range(numX):
                    outz=0
                    for k in range(kiters):
                        if(k>=numKernels):
                            value=0
                        else:   
                            #print("array f is " +str(f)+ " y is "+str(y)+" x is "+str(x)+" k is "+str(k))
                            value=array[f,y,x,k]
                        #print("value is "+str(value))

                        values=(values<<args.weights) | (value&((1<<int(args.weights))-1))

                        if((k+1)%math.floor(64/int(args.weights))==0 or k==kiters-1):
                            #if(k%math.floor(64/int(args.weights))==0):
                                #for
                            #print("out array f is " +str(f)+ " y is "+str(y)+" x is "+str(x)+" outz is "+str(outz))
                            #print("values is "+str(values))
                            outarray[f,y,x,outz]= values<<remainder
                            outz=outz+1
                            values=0


        weightobj = open("output/"+format(filecounter, '04d')+key+".bin",mode='wb')
        filecounter=filecounter+1
        outarray.tofile(weightobj)
        weightobj.close()
        #print("size is " + str(numFilters*numY*numX*zvalues))
        #print("memAdress is " + str(memAddress))



        #input("ENTER")

    if("bias" in key):
        #descriptionObj.write("model."+key.replace(".bias","",1)+".int_bias()\n")
        array=eval("model."+key.replace(".bias","",1)+".int_bias()")
        array = array.cpu().numpy().astype(int)
        #print(array)
        descriptionObj.write("Layer Name = "+ key+"\n")
        print("Layer Name = "+ key)
        scale=eval("model."+key.replace("bias","quant_bias_scale().cpu().detach().numpy()[0]",1)+"\n")
        descriptionObj.write("Scale = "+ str(scale)+"\n")
        descriptionObj.write("Shift "+ str(abs(math.log(scale,2)))+ " positions to the right\n")
        descriptionObj.write("Address is " + hex(memAddress)+"\n\n")
        
        #print("Scale = " + str(eval("model."+key.replace("bias","quant_bias_scale()",1))))
        #kiters=math.ceil(numKernels/math.floor(64/int(args.weights)))*math.floor(64/int(args.weights))
        numBias=array.shape[0]
        numIters=math.ceil(numBias/math.floor(64/8))*math.floor(64/8)
        memAddress=memAddress+((math.ceil(numBias/math.floor(64/8)))*8)
        #print("numBias is "+ str(numBias))
        #print("numiters is "+ str(numIters))
        index=0
        values=0
        outarray = np.zeros((math.ceil(numBias/math.floor(64/8))),dtype=np.int64)
        for i in range(numIters):
            if(i>=numBias):
                value=0
            else:   
                value=array[i]
            #print(value)
            values=(values<<8) | (value&((1<<8)-1))
            if((i+1)%math.floor(64/8)==0 or i==numIters-1):
                #print("values is " + str(values) )
                outarray[index]= values
                index=index+1
                values=0

        biasobj = open("output/"+format(filecounter, '04d')+key+".bin",mode='wb')
        filecounter=filecounter+1
        outarray.tofile(biasobj)
        biasobj.close()
        #print("size is " + str((math.ceil(numBias/math.floor(64/8)))))
        #print("memAdress is " + str(memAddress))




        #input("ENTER")

    if("act_quant" in key):
        layerstr=key[:key.rfind("act_quant")-1]
        print("Layer Name = "+ layerstr)
        descriptionObj.write("Layer Name = "+ layerstr+"\n")
        descriptionObj.write("Scale = " + str(eval("model."+layerstr+".quant_act_scale().cpu().detach().numpy()"))+"\n")
        scale=eval("model."+layerstr+".quant_act_scale().cpu().detach().numpy()")
        descriptionObj.write("Shift "+ str(abs(math.log(scale,2)))+ " positions to the right\n\n")
        #input("ENTER")


