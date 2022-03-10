import argparse
import os


parser = argparse.ArgumentParser(description='Weights Extractor')
parser.add_argument('--input', default="./in.py", type=str, help='input file')
parser.add_argument('--output', default="./out.py", type=str, help='output file')
parser.add_argument('--engine', default="./Example/quantizers.py", type=str, help='output file')
parser.add_argument('--weightsEngine', default="CustomWeightQuant", type=str, help='output file')
parser.add_argument('--activationsEngine', default="CustomActQuant", type=str, help='output file')
args = parser.parse_args()


						
with open(args.output, 'w') as outfile:
	outfile.write("#This file was generated with brevitasConverter.py\n")
	outfile.write("import brevitas.nn as qnn\nfrom brevitas.quant import Int8Bias as BiasQuant\n\n")
	outfile.write("#Engine declaration\n")
	with open(args.engine,'r') as enginefile:
		for line in enginefile:
			outfile.write(line)
	outfile.write("\n#Global Variables\n\n#End of Engine declaration\n\n")
	outfile.write("weightBitWidth=8\nactivationBitWidth=8\n\n")

	with open(args.input,'r') as file:
		for line in file:
			if(".__init__()" in line):
				line=line+"        global weightBitWidth\n        global activationBitWidth\n"
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
			if("nn.Linear" in line):
				line=line.replace("nn.Linear","qnn.QuantLinear",1)

				lastParentheses=line.rfind(")")
				if("bias=" in line):
					line=line[:lastParentheses] + ", weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant="+ args.weightsEngine +", return_quant_tensor=True"+line[lastParentheses:]
				else:
					line=line[:lastParentheses] + ", bias=True, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant="+ args.weightsEngine +", return_quant_tensor=True"+line[lastParentheses:]
			if("nn.AvgPool2d" in line):
				line=line.replace("nn.AvgPool2d","qnn.QuantAvgPool2d",1)

				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ",bit_width=activationBitWidth, return_quant_tensor=True"+line[lastParentheses:]
			if("nn.MaxPool2d" in line):
				line=line.replace("nn.MaxPool2d","qnn.QuantMaxPool2d",1)

				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", return_quant_tensor=True"+line[lastParentheses:]




			outfile.write(line)
	outfile.write("\n\nclass setBitWidths():\n    def __init__(self,weight,activation):\n        global weightBitWidth\n        global activationBitWidth\n        weightBitWidth=weight\n        activationBitWidth=activation")
outfile.close()
print("Conversion done.\n\
Please include the following modifications on the output file:\n\
Include the following as the first layer:\n\
	\"self.quant_inp = qnn.QuantIdentity(bit_width=activationBitWidth, return_quant_tensor=True)\"\n\
Turn the return_quant_tensor parameter of the last layer from True to False\n\
On the training script define the bit_width of the weights and activations by calling \"setBitWidths(weights,activations)\" before you call the model")