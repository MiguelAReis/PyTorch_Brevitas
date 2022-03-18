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

'''ResNetQuant in PyTorch.

For Pre-activation ResNet, see 'preact_resnetQuant.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from dependencies import value



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = qnn.QuantConv2d(planes, planes, kernel_size=3,stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def setBitWidths(weight,activation):
        global weightBitWidth
        global activationBitWidth
        weightBitWidth=weight
        activationBitWidth=activation

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = qnn.QuantConv2d(in_planes, planes, kernel_size=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = qnn.QuantConv2d(planes, planes, kernel_size=3,stride=stride, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = qnn.QuantConv2d(planes, self.expansion * planes, kernel_size=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu1 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                qnn.QuantConv2d(in_planes, self.expansion*planes,kernel_size=1, stride=stride, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def setBitWidths(weight,activation):
        global weightBitWidth
        global activationBitWidth
        weightBitWidth=weight
        activationBitWidth=activation

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNetQuant(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetQuant, self).__init__()
        global weightBitWidth
        global activationBitWidth

        self.quant_inp = qnn.QuantIdentity(bit_width=8, act_quant=CustomActQuant, return_quant_tensor=True)
        self.in_planes = 64

        self.conv1 = qnn.QuantConv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = qnn.QuantLinear(512*block.expansion, num_classes, bias=True, weight_bit_width=weightBitWidth, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=False)
        self.relu = qnn.QuantReLU(bit_width=activationBitWidth, return_quant_tensor=True, act_quant=CustomActQuant) if weightBitWidth not in (1,2) else qnn.QuantIdentity(bit_width=activationBitWidth, act_quant=CustomActQuant, return_quant_tensor=True)
        self.avg_pool2d = qnn.QuantAvgPool2d(4, bit_width=activationBitWidth, return_quant_tensor=True)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def setBitWidths(weight,activation):
        global weightBitWidth
        global activationBitWidth
        weightBitWidth=weight
        activationBitWidth=activation

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(self.quant_inp(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNetQuant18():
    return ResNetQuant(BasicBlock, [2, 2, 2, 2])


def ResNetQuant34():
    return ResNetQuant(BasicBlock, [3, 4, 6, 3])


def ResNetQuant50():
    return ResNetQuant(Bottleneck, [3, 4, 6, 3])


def ResNetQuant101():
    return ResNetQuant(Bottleneck, [3, 4, 23, 3])


def ResNetQuant152():
    return ResNetQuant(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNetQuant18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
