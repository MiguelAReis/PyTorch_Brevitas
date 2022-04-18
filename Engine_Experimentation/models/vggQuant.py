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

 '''VGGQuant11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
	'VGGQuant11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGGQuant13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGGQuant16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGGQuant19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGQuant(nn.Module):
	def __init__(self, vggQuant_name):
		super(VGGQuant, self).__init__()
		global weightBitWidth
		global activationBitWidth

		self.imageQuant = qnn.QuantIdentity(bit_width=8, act_quant=CustomActQuant, return_quant_tensor=True)
		self.features = self._make_layers(cfg[vggQuant_name])
		self.classifier = qnn.QuantLinear(512, 10, bias=True, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)

	def setBitWidths(weight,activation):
		global weightBitWidth
		global activationBitWidth
		weightBitWidth=weight
		activationBitWidth=activation

	def forward(self, x):
		out = self.features(self.imageQuant(x))
		out = out.view(out.size(0), -1)
		out = self.classifier(out)
		return out

	def _make_layers(self, cfg):
		layers = []
		in_channels = 3
		for x in cfg:
			if x == 'M':
				layers += [qnn.QuantMaxPool2d(kernel_size=2, stride=2, return_quant_tensor=True)]
			else:
				layers += [qnn.QuantConv2d(in_channels, x, kernel_size=3, padding=1, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
						   nn.BatchNorm2d(x),
						   qnn.QuantReLU(inplace=True, bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)]
				in_channels = x
		layers += [qnn.QuantAvgPool2d(kernel_size=1, stride=1, bit_width=activationBitWidth, return_quant_tensor=True)]
		return nn.Sequential(*layers)


def test():
	net = VGG('VGGQuant11')
	x = torch.randn(2,3,32,32)
	y = net(x)
	print(y.size())
