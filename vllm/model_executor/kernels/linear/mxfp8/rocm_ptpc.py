# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 linear for gfx942 as an aiter per-token x per-channel FP8 GEMM.

gfx942 has no microscaling matrix cores, so the weight is requantized at load
to one power-of-two scale per output row (see aiter's mxfp8_to_per_row_fp8)
and activations are quantized per token. The GEMM is hipBLASLt's, with online
tuning, for every shape it takes, else aiter's for shapes aiter has tuned; other
shapes stay on the BF16 emulation.
"""

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

from .emulation import EmulationMxfp8LinearKernel
from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class _HipbMMKernel(AiterHipbMMPerTokenFp8ScaledMMLinearKernel):
    """vLLM's hipBLASLt per-token FP8 kernel, on by default for these linears.
    VLLM_ROCM_USE_AITER_LINEAR_HIPBMM, which its base requires, would move every
    other per-token FP8 layer as well."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        import aiter

        if not hasattr(aiter, "hipb_mm"):
            return False, "requires aiter hipb_mm support."
        return True, None


# hipBLASLt first: it takes every shape with K divisible by 64 and is the faster
# GEMM for these linears. aiter's preshuffled GEMM takes the shapes aiter has
# tuned.
_PER_TOKEN_KERNELS: tuple[type[FP8ScaledMMLinearKernel], ...] = (
    _HipbMMKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)


def _per_token_config(c: Mxfp8LinearLayerConfig) -> FP8ScaledMMLinearLayerConfig:
    return FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticChannelSym,
        activation_quant_key=kFp8DynamicTokenSym,
        weight_shape=c.weight_shape,
        input_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
    )


def _is_disabled(kernel: type[FP8ScaledMMLinearKernel]) -> bool:
    # by the names in vLLM's kernel lists, so _HipbMMKernel counts as its base
    return any(k.__name__ in envs.VLLM_DISABLED_KERNELS for k in kernel.__mro__)


def _per_token_kernel(
    c: Mxfp8LinearLayerConfig,
) -> type[FP8ScaledMMLinearKernel] | None:
    config = _per_token_config(c)
    for kernel in _PER_TOKEN_KERNELS:
        if (
            not _is_disabled(kernel)
            and kernel.is_supported()[0]
            and kernel.can_implement(config)[0]
        ):
            return kernel
    return None


def has_aiter_requant() -> bool:
    try:
        from aiter.ops.quant import mxfp8_to_per_row_fp8  # noqa: F401
    except ImportError:
        return False
    return True


def _keeps_mxfp8_layout(layer: torch.nn.Module) -> bool:
    # DeepSeek V4/V4.1 wo_a: the attention reads its weight and E8M0 scale
    # directly (a grouped einsum on ROCm), so it keeps the emulation layout.
    return getattr(layer, "bmm_batch_size", None) is not None


class RocmPtpcMxfp8LinearKernel(Mxfp8LinearKernel):
    """gfx942 MXFP8 linear on per-row requantized weights and an aiter
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
        if not has_aiter_requant():
            return False, "requires an aiter with ops.quant.mxfp8_to_per_row_fp8"
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        if _per_token_kernel(c) is None:
            return False, (
                "neither hipBLASLt (K divisible by 64) nor a tuned aiter GEMM "
                "takes this shape"
            )
        return True, None

    def __init__(self, c: Mxfp8LinearLayerConfig) -> None:
        super().__init__(c)
        per_token_kernel = _per_token_kernel(c)
        assert per_token_kernel is not None
        self.per_token = per_token_kernel(
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
        # The per-token kernels take the [K, N] weight every FP8 method leaves
        # behind and preshuffle it.
        replace_parameter(layer, "weight", weight.t())
        replace_parameter(layer, "weight_scale", weight_scale)
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
