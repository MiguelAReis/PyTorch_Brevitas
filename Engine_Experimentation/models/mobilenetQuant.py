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
        #    return QuantType.TERNARY
        else:
            return QuantType.INT

class CustomWeightQuant(CustomQuant,WeightQuantSolver):
    scaling_const = 1.0        

class CustomActQuant(CustomQuant, ActQuantSolver):
    min_val = 0
    max_val = 10

#Global Variables

#End of Engine declaration

'''MobileNetQuant in PyTorch.

See the paper "MobileNetQuants: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = qnn.QuantConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu1 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

    def setBitWidths(weight,activation):
        global weightBitWidth
        global activationBitWidth
        weightBitWidth=weight
        activationBitWidth=activation

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return out


class MobileNetQuant(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetQuant, self).__init__()
        global weightBitWidth
        global activationBitWidth

        self.quant_inp = qnn.QuantIdentity(bit_width=8, act_quant=CustomActQuant, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = qnn.QuantLinear(1024, num_classes, bias=True, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
        self.relu1 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.avg_pool= qnn.QuantAvgPool2d(2, bit_width=activationBitWidth, return_quant_tensor=True)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def setBitWidths(weight,activation):
        global weightBitWidth
        global activationBitWidth
        weightBitWidth=weight
        activationBitWidth=activation

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(self.quant_inp(x))))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test():
    net = MobileNetQuant()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()
