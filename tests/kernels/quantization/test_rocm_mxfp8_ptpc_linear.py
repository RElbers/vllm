# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 MXFP8 linear on per-row requantized weights and aiter's per-token
FP8 GEMM (RocmPtpcMxfp8LinearKernel)."""

import importlib.util

import pytest
import torch
from torch.nn.parameter import Parameter

from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.kernels.linear import init_mxfp8_linear_kernel
from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mxfp8.rocm_ptpc import (
    RocmPtpcMxfp8LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.platforms import current_platform


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


pytestmark = [
    pytest.mark.skipif(
        not (_on_gfx942() and importlib.util.find_spec("aiter") is not None),
        reason="Requires ROCm gfx942 + aiter",
    ),
    pytest.mark.usefixtures("default_vllm_config"),
]

# Dense MXFP8 linears of a DeepSeek V4.1 Flash TP4 rank with tuned aiter rows.
TUNED_SHAPES = [
    (8192, 1280),
    (5120, 2048),
    (1792, 5120),
    (1152, 5120),
    (5120, 576),
    (4096, 1280),
]
UNTUNED_SHAPE = (528, 1056)


def _set_flags(monkeypatch: pytest.MonkeyPatch, ptpc: str) -> None:
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MXFP8_PTPC", ptpc)
    rocm_aiter_ops.refresh_env_variables()


@pytest.fixture
def enable_ptpc(monkeypatch: pytest.MonkeyPatch):
    _set_flags(monkeypatch, "1")
    yield
    rocm_aiter_ops.refresh_env_variables()


def _make_layer(n: int, k: int) -> torch.nn.Module:
    """An MXFP8 layer as vLLM loads it: 32x32 E8M0 blocks expanded per row."""
    layer = torch.nn.Module()
    # values span the e4m3fn range like MX-quantized blocks
    weight = (torch.randn(n, k, device="cuda") * 100).clamp(-448, 448)
    blocks = torch.randint(118, 122, (n // 32, k // 32), device="cuda")
    layer.weight = Parameter(weight.to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_scale = Parameter(
        blocks.to(torch.uint8).repeat_interleave(32, dim=0), requires_grad=False
    )
    return layer


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("weight_shape", [TUNED_SHAPES[1], UNTUNED_SHAPE])
@pytest.mark.parametrize("ptpc", ["0", "1"])
def test_selection(
    monkeypatch: pytest.MonkeyPatch, weight_shape: tuple[int, int], ptpc: str
):
    _set_flags(monkeypatch, ptpc)
    kernel = init_mxfp8_linear_kernel(weight_shape=weight_shape, bmm_batch_size=None)
    expected = (
        RocmPtpcMxfp8LinearKernel
        if ptpc == "1" and weight_shape != UNTUNED_SHAPE
        else EmulationMxfp8LinearKernel
    )
    assert type(kernel) is expected
    rocm_aiter_ops.refresh_env_variables()


@pytest.mark.parametrize("weight_shape", TUNED_SHAPES)
@pytest.mark.parametrize("x_shape", [(1,), (300,), (8192,), (3, 100)])
@torch.inference_mode()
def test_matches_mxfp8_reference(
    enable_ptpc, weight_shape: tuple[int, int], x_shape: tuple[int, ...]
):
    torch.manual_seed(0)
    n, k = weight_shape
    layer = _make_layer(n, k)
    w_ref = dequant_mxfp8_to_bf16(layer.weight.data, layer.weight_scale.data).float()
    kernel = RocmPtpcMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape))
    kernel.process_weights_after_loading(layer)
    assert layer.weight.dtype == current_platform.fp8_dtype()
    assert layer.weight_scale.shape == (n, 1)

    x = torch.randn(*x_shape, k, dtype=torch.bfloat16, device="cuda")
    out = kernel.apply_weights(layer, x)

    assert out.shape == (*x_shape, n) and out.dtype == torch.bfloat16
    # per-token FP8 activations are the only lossy step
    assert _rel_err(out, x.float() @ w_ref.t()) < 0.05


@torch.inference_mode()
def test_wo_a_keeps_emulation_layout(enable_ptpc):
    torch.manual_seed(0)
    n, k = TUNED_SHAPES[1]
    layer = _make_layer(n, k)
    layer.bmm_batch_size = 2
    w_ref = dequant_mxfp8_to_bf16(layer.weight.data, layer.weight_scale.data)
    kernel = RocmPtpcMxfp8LinearKernel(Mxfp8LinearLayerConfig((n, k)))
    kernel.process_weights_after_loading(layer)
    assert layer.weight.dtype == torch.bfloat16
    assert torch.equal(layer.weight.data, w_ref)

    x = torch.randn(16, k, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(kernel.apply_weights(layer, x), x @ w_ref.t())
