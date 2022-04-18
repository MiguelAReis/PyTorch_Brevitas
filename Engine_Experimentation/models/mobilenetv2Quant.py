#This file was generated with brevitasConverter.py
weightBitWidth=8
activationBitWidth=8

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant

#Engine declaration
from brevitas.inject import ExtendedInjector
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject.enum import ScalingImplType, StatsOp, RestrictValueType
from dependencies import value

class CustomQuant(ExtendedInjector):
	bit_width_impl_type = BitWidthImplType.CONST
	scaling_impl_type = ScalingImplType.CONST
	restrict_scaling_type = RestrictValueType.POWER_OF_TWO
	zero_point_impl = ZeroZeroPoint
	float_to_int_impl_type = FloatToIntImplType.ROUND
	scaling_impl_type = ScalingImplType.STATS
	scaling_stats_op = StatsOp.MAX
	scaling_per_output_channel = False
	bit_width = None
	narrow_range = True
	signed = True
	
	@value
	def quant_type():
		global weightBitWidth
		if weightBitWidth == 1:
			return QuantType.BINARY
		#elif  weightBitWidth ==2:
		#	return QuantType.TERNARY
		else:
			return QuantType.INT

class CustomWeightQuant(CustomQuant,WeightQuantSolver):
	scaling_const = 1.0		

class CustomActQuant(CustomQuant, ActQuantSolver):
	min_val = 0
	max_val = 10

#Global Variables

#End of Engine declaration

'''MobileNetV2Quant in PyTorch.
See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
	'''expand + depthwise + pointwise'''
	def __init__(self, in_planes, out_planes, expansion, stride):
		super(Block, self).__init__()
		self.stride = stride

		planes = expansion * in_planes
		self.conv1 = qnn.QuantConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = qnn.QuantConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = qnn.QuantConv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn3 = nn.BatchNorm2d(out_planes)

		self.shortcut = nn.Sequential()
		if stride == 1 and in_planes != out_planes:
			self.shortcut = nn.Sequential(
				qnn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
				nn.BatchNorm2d(out_planes),
			)


	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out = out + self.shortcut(x) if self.stride==1 else out
		return out


class MobileNetV2Quant(nn.Module):
	# (expansion, out_planes, num_blocks, stride)
	cfg = [(1,  16, 1, 1),
		   (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
		   (6,  32, 3, 2),
		   (6,  64, 4, 2),
		   (6,  96, 3, 1),
		   (6, 160, 3, 2),
		   (6, 320, 1, 1)]

	def __init__(self, num_classes=10):
		super(MobileNetV2Quant, self).__init__()
		global weightBitWidth
		global activationBitWidth

		self.imageQuant = qnn.QuantIdentity(bit_width=8, act_quant=CustomActQuant, return_quant_tensor=True)
		# NOTE: change conv1 stride 2 -> 1 for CIFAR10
		self.conv1 = qnn.QuantConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn1 = nn.BatchNorm2d(32)
		self.relu1 =qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.layers = self._make_layers(in_planes=32)
		self.conv2 = qnn.QuantConv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
		self.bn2 = nn.BatchNorm2d(1280)
		self.relu2 =qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
		self.avgpool=qnn.QuantAvgPool2d(4, bit_width=activationBitWidth, return_quant_tensor=True)
		self.linear = qnn.QuantLinear(1280, num_classes, bias=True, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)

	def _make_layers(self, in_planes):
		layers = []
		for expansion, out_planes, num_blocks, stride in self.cfg:
			strides = [stride] + [1]*(num_blocks-1)
			for stride in strides:
				layers.append(Block(in_planes, out_planes, expansion, stride))
				in_planes = out_planes
		return nn.Sequential(*layers)

	def setBitWidths(weight,activation):
		global weightBitWidth
		global activationBitWidth
		weightBitWidth=weight
		activationBitWidth=activation

	def forward(self, x):
		out = self.conv1(self.imageQuant(x))
		out = self.bn1(x)
		out = self.relu1(x)
		out = self.layers(out)
		out = self.conv2(x)
		out = self.bn2(x)
		out = self.relu2(x)
		# NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
		out = self.avgpool(out)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def test():
	net = MobileNetV2Quant()
	x = torch.randn(2,3,32,32)
	y = net(x)
	print(y.size())

# test()