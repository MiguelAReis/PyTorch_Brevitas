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
	outfile.write("\n#End of Engine declaration\n\n")

	with open(args.input,'r') as file:
		for line in file:
			if("def __init__(" in line):
				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", weights=8, activations=8" + line[lastParentheses:]
			if("nn.Conv2d" in line):
				line=line.replace("nn.Conv2d","qnn.QuantConv2d",1)

				lastParentheses=line.rfind(")")
				line=line[:lastParentheses] + ", weight_bit_width=weights, bias_quant=BiasQuant, weight_quant="+ args.weightsEngine +", return_quant_tensor=True"+line[lastParentheses:]

			if("nn.ReLU" in line):
				line=line.replace("nn.ReLU","qnn.QuantReLU",1)

				lastParentheses=line.rfind(")")
				if(line[lastParentheses-1]=='('):
					line=line[:lastParentheses] + "bit_width=activations, return_quant_tensor=True, act_quant="+args.activationsEngine+line[lastParentheses:]
				else:
					line=line[:lastParentheses] + ", bit_width=activations, return_quant_tensor=True, act_quant="+args.activationsEngine+line[lastParentheses:]
			outfile.write(line)

outfile.close()