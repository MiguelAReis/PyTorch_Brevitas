import argparse
import os


parser = argparse.ArgumentParser(description='Converts a model description file into a quantized model with brevitas functions')
parser.add_argument('--input', default="./in.py", type=str, help='Input net description file e.g. lenet.py')
parser.add_argument('--output', default="./out.py", type=str, help='Output net description file e.g. lenetQuant.py')
parser.add_argument('--engine', default="./Example/quantizers.py", type=str, help='Quantization engine file e.g. Example/quantizers.py')
parser.add_argument('--weightsEngine', default="CustomWeightQuant", type=str, help='The name of the class for the engine used in weight quantizations e.g. CustomWeightQuant')
parser.add_argument('--activationsEngine', default="CustomActQuant", type=str, help='The name of the class for the engine used in Activations quantizations e.g. CustomActQuant')
parser.add_argument('--inputBitWidth', default=8, type=int, help="The bit width of the input images")
args = parser.parse_args()

firstlayer=False
forward=False
prevLn=""
lastLayer=""
						
with open(args.output, 'w') as outfile:
	outfile.write("#This file was generated with brevitasConverter.py\n")
	outfile.write("weightBitWidth=8\nactivationBitWidth=8\n\n")
	outfile.write("import brevitas.nn as qnn\nfrom brevitas.quant import Int8Bias as BiasQuant\n\n")
	outfile.write("#Engine declaration\n")
	with open(args.engine,'r') as enginefile:
		for line in enginefile:
			outfile.write(line)
	outfile.write("\n#Global Variables\n\n#End of Engine declaration\n\n")

	netname=args.input[args.input.rfind("/")+1:].replace(".py","",1)
	layerInput="BLANK"
	out="BLANK"


	with open(args.input,'r') as file:
		for line in file:
			if(netname.casefold() in line.casefold()):
				namePos=line.casefold().rfind(netname.casefold())
				quantNetName=line[namePos:namePos+len(netname)]+"Quant"
				line = line[:namePos+len(netname)]+"Quant"+line[namePos+len(netname):]
				if(("super("+quantNetName) in line):
					firstlayer=True
					forward=True

			if(".__init__()" in line and firstlayer):
				line=line+"        global weightBitWidth\n        global activationBitWidth\n\n        self.imageQuant = qnn.QuantIdentity(bit_width="+str(args.inputBitWidth)+", act_quant=CustomActQuant, return_quant_tensor=True)\n"

			if("def forward" in line and firstlayer):
				lastParentheses=line.rfind(")")
				layerInput=line[line[:lastParentheses].rfind(" ")+1:lastParentheses]
				line="    def setBitWidths(weight,activation):\n        global weightBitWidth\n        global activationBitWidth\n        weightBitWidth=weight\n        activationBitWidth=activation\n\n"+line
			if(("("+layerInput+")") in line and firstlayer):
				line=line.replace(layerInput,"self.imageQuant("+layerInput+")",1)
				firstlayer=False
				out=line[:line.find("=")].replace("	","").replace(" ","")
			if(("return "+out) in line):
				lastLayer=prevLn
				forward=False
				
			if("nn.Conv2d" in line):
				line=line.replace("nn.Conv2d","qnn.QuantConv2d",1)

				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant="+ args.weightsEngine +", return_quant_tensor=True"+line[lastParentheses:]

			if("nn.ReLU" in line):
				line=line.replace("nn.ReLU","qnn.QuantReLU",1)

				lastParentheses=line.rfind(")")
				if(line[lastParentheses-1]=='('):
					line=line[:lastParentheses] + "bit_width=activationBitWidth, return_quant_tensor=True, act_quant="+args.activationsEngine+line[lastParentheses:]
				else:
					line=line[:lastParentheses] + ", bit_width=activationBitWidth, return_quant_tensor=True, act_quant="+args.activationsEngine+line[lastParentheses:]
				
				lastParentheses=line.rfind(")")
				line=line[:lastParentheses+1] +" if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)"+line[lastParentheses+1:]
			if("nn.Linear" in line):
				line=line.replace("nn.Linear","qnn.QuantLinear",1)

				lastParentheses=line.rfind(")")
				if("bias=" in line):
					line=line[:lastParentheses] + ", weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant="+ args.weightsEngine +", return_quant_tensor=True"+line[lastParentheses:]
				else:
					line=line[:lastParentheses] + ", bias=True, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant="+ args.weightsEngine +", return_quant_tensor=True"+line[lastParentheses:]

			if("nn.AdaptiveAvgPool2d" in line):
				line=line.replace("nn.AdaptiveAvgPool2d","qnn.QuantAdaptiveAvgPool2d",1)

				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", bit_width=activationBitWidth, return_quant_tensor=True"+line[lastParentheses:]

			if("nn.Dropout" in line):
				line=line.replace("nn.Dropout","qnn.QuantDropout",1)
				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", return_quant_tensor=True"+line[lastParentheses:]

			if("nn.AvgPool2d" in line):
				line=line.replace("nn.AvgPool2d","qnn.QuantAvgPool2d",1)

				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", bit_width=activationBitWidth, return_quant_tensor=True"+line[lastParentheses:]
			if("nn.MaxPool2d" in line):
				line=line.replace("nn.MaxPool2d","qnn.QuantMaxPool2d",1)

				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", return_quant_tensor=True"+line[lastParentheses:]
			'''
			if("nn.BatchNorm2d" in line):
				line=line.replace("nn.BatchNorm2d","qnn.BatchNorm2dToQuantScaleBias",1)

				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", bit_width=activationBitWidth, return_quant_tensor=True"+line[lastParentheses:]
			'''




			prevLn= line
			outfile.write(line)
outfile.close()



lastLayer=lastLayer[lastLayer.find("self."):lastLayer.find("(")]
with open(args.output, 'r') as outfile:
	replacement = ""
	# using the for loop
	for line in outfile:
		if(("super("+quantNetName) in line):
			forward=True
		if(lastLayer in line and forward):
			line=line.replace("return_quant_tensor=True","return_quant_tensor=False",1)
			forward=False
		replacement = replacement + line
	outfile.close()

outfile = open(args.output, "w")
outfile.write(replacement)
outfile.close()


print("Conversion done.\n\
New net name changed to " + quantNetName +"\n\
On the training script define the bit_width of the weights and activations by calling \""+quantNetName+".setBitWidths(weights,activations)\" before you call the model")
