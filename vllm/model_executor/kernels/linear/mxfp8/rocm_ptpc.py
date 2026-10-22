# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 linear for gfx942 as an aiter per-token x per-channel FP8 GEMM.

gfx942 has no microscaling matrix cores. At load, the weight's E8M0 block
scales are folded into one power-of-two scale per output row: blocks are
shifted by whole exponents, so only values pushed into the subnormals round.
Activations are quantized per token and the GEMM is aiter's preshuffled
per-token FP8 GEMM, which ``can_implement`` admits only for shapes aiter has
tuned; every other shape stays on the BF16 emulation.
"""

import torch
from torch.nn.parameter import Parameter

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.platforms import current_platform

from .emulation import EmulationMxfp8LinearKernel
from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig

_PerTokenKernel = AiterPreshuffledPerTokenFp8ScaledMMLinearKernel


def _per_token_config(c: Mxfp8LinearLayerConfig) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=c.weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )


def _keeps_mxfp8_layout(layer: torch.nn.Module) -> bool:
    # DeepSeek V4/V4.1 wo_a: the attention reads its weight and E8M0 scale
    # directly (a grouped einsum on ROCm), so it keeps the emulation layout.
    return getattr(layer, "bmm_batch_size", None) is not None


class RocmPtpcMxfp8LinearKernel(Mxfp8LinearKernel):
    """gfx942 MXFP8 linear on per-row requantized weights and aiter's
    per-token FP8 GEMM."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not ROCm"
        if not rocm_aiter_ops.is_mxfp8_ptpc_enabled():
            return False, (
                "requires gfx942, VLLM_ROCM_USE_AITER=1 and "
                "VLLM_ROCM_USE_AITER_MXFP8_PTPC=1"
            )
        return _PerTokenKernel.is_supported(compute_capability)

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if c.bmm_batch_size is not None:
            return False, "batched MXFP8 GEMMs are not supported"
        return _PerTokenKernel.can_implement(_per_token_config(c))

    def __init__(self, c: Mxfp8LinearLayerConfig) -> None:
        super().__init__(c)
        self.per_token = _PerTokenKernel(
            _per_token_config(c),
            layer_param_names=[
                "weight",
                "weight_scale",
                "input_scale",
                "input_scale_ub",
            ],
        )
        self.emulation = EmulationMxfp8LinearKernel(c)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _keeps_mxfp8_layout(layer):
            self.emulation.process_weights_after_loading(layer)
            return
        from aiter.ops.quant import mxfp8_to_per_row_fp8

        N, K = layer.weight.shape
        weight, weight_scale = mxfp8_to_per_row_fp8(
            layer.weight.data,
            layer.weight_scale.data[:N, : K // MXFP8_BLOCK_SIZE],
            quant_dtype=current_platform.fp8_dtype(),
        )
        # The per-token kernel takes the [K, N] weight every FP8 method leaves
        # behind and preshuffles it.
        layer.weight = Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        self.per_token.process_weights_after_loading(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if _keeps_mxfp8_layout(layer):
            return self.emulation.apply_weights(layer, x, bias)
        out = self.per_token.apply_weights(layer, x, bias)
        return out.view(*x.shape[:-1], out.shape[-1])
