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
	restrict_scaling_type = RestrictValueType.POWER_OF_TWO
	zero_point_impl = ZeroZeroPoint
	float_to_int_impl_type = FloatToIntImplType.ROUND
	scaling_impl_type = ScalingImplType.STATS
	scaling_stats_op = StatsOp.MAX
	scaling_per_output_channel = False
	bit_width = None
	narrow_range = False
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
    signed=False
    float_to_int_impl_type = FloatToIntImplType.FLOOR

class CustomSignedActQuant(CustomQuant, ActQuantSolver):
    signed=True
    float_to_int_impl_type = FloatToIntImplType.FLOOR
