#This file was generated with brevitasConverter.py
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
    quant_type = QuantType.INT

class CustomWeightQuant(CustomQuant,WeightQuantSolver):
    scaling_const = 1.0        

class CustomActQuant(CustomQuant, ActQuantSolver):
    min_val = -1.0
    max_val = 1.0-1.0/64

#End of Engine declaration

'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, weights=8, activations=8):
        super(LeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, weight_bit_width=weights, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.relu1  = qnn.QuantReLU(bit_width=activations, return_quant_tensor=True, act_quant=CustomActQuant)
        self.conv2 = qnn.QuantConv2d(6, 16, 5, weight_bit_width=weights, bias_quant=BiasQuant, weight_quant=CustomWeightQuant, return_quant_tensor=True)
        self.relu2  = qnn.QuantReLU(bit_width=activations, return_quant_tensor=True, act_quant=CustomActQuant)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.relu3  = qnn.QuantReLU(bit_width=activations, return_quant_tensor=True, act_quant=CustomActQuant)
        self.fc2   = nn.Linear(120, 84)
        self.relu4  = qnn.QuantReLU(bit_width=activations, return_quant_tensor=True, act_quant=CustomActQuant)
        self.fc3   = nn.Linear(84, 10)


    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
