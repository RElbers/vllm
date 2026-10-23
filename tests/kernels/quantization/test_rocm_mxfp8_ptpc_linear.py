# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx942 MXFP8 linear on per-row requantized weights and aiter's per-token
FP8 GEMM (RocmPtpcMxfp8LinearKernel)."""

import functools
from types import SimpleNamespace

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
    has_aiter_requant,
)
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    dequant_mxfp8_to_bf16,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.platforms import current_platform


def _on_gfx942() -> bool:
    if not current_platform.is_rocm():
        return False
    from vllm.platforms.rocm import on_gfx942

    return on_gfx942()


pytestmark = [
    pytest.mark.skipif(
        not (_on_gfx942() and has_aiter_requant()),
        reason="Requires ROCm gfx942 + aiter with mxfp8_to_per_row_fp8",
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
UNTUNED_SHAPE = (544, 1088)
# hipBLASLt's preshuffled FP8 GEMM is wrong for K % 64 != 0, so nothing takes it
K_NOT_64_SHAPE = (544, 1056)


def _set_flags(monkeypatch: pytest.MonkeyPatch, ptpc: str, gemm: str) -> None:
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_LINEAR", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MXFP8_PTPC", ptpc)
    if gemm == "aiter":
        monkeypatch.setenv(
            "VLLM_DISABLED_KERNELS", "AiterHipbMMPerTokenFp8ScaledMMLinearKernel"
        )
    rocm_aiter_ops.refresh_env_variables()


@pytest.fixture
def set_flags(monkeypatch: pytest.MonkeyPatch):
    yield functools.partial(_set_flags, monkeypatch)
    # restore the environment first, so that the refresh reads the real flags
    monkeypatch.undo()
    rocm_aiter_ops.refresh_env_variables()


@pytest.fixture(params=["aiter", "hipblaslt"])
def enable_ptpc(set_flags, request):
    set_flags("1", request.param)
    return request.param


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


@pytest.mark.parametrize(
    "weight_shape", [TUNED_SHAPES[1], UNTUNED_SHAPE, K_NOT_64_SHAPE]
)
@pytest.mark.parametrize("ptpc", ["0", "1"])
@pytest.mark.parametrize("gemm", ["aiter", "hipblaslt"])
def test_selection(
    set_flags,
    weight_shape: tuple[int, int],
    ptpc: str,
    gemm: str,
):
    set_flags(ptpc, gemm)
    kernel = init_mxfp8_linear_kernel(weight_shape=weight_shape, bmm_batch_size=None)
    # hipBLASLt takes any shape; aiter's own GEMM only tuned ones
    takes_shape = weight_shape == TUNED_SHAPES[1] or (
        gemm == "hipblaslt" and weight_shape == UNTUNED_SHAPE
    )
    expected = (
        RocmPtpcMxfp8LinearKernel
        if ptpc == "1" and takes_shape
        else EmulationMxfp8LinearKernel
    )
    assert type(kernel) is expected


@pytest.mark.parametrize("weight_shape", [*TUNED_SHAPES, UNTUNED_SHAPE])
@pytest.mark.parametrize("x_shape", [(1,), (300,), (8192,), (3, 100)])
@torch.inference_mode()
def test_matches_mxfp8_reference(
    enable_ptpc, weight_shape: tuple[int, int], x_shape: tuple[int, ...]
):
    if weight_shape == UNTUNED_SHAPE and enable_ptpc == "aiter":
        pytest.skip("aiter's own GEMM takes only tuned shapes")
    torch.manual_seed(0)
    n, k = weight_shape
    layer = _make_layer(n, k)
    w_ref = dequant_mxfp8_to_bf16(layer.weight.data, layer.weight_scale.data).float()
    kernel = RocmPtpcMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape))
    gemm = {
        "aiter": AiterPreshuffledPerTokenFp8ScaledMMLinearKernel,
        "hipblaslt": AiterHipbMMPerTokenFp8ScaledMMLinearKernel,
    }[enable_ptpc]
    assert isinstance(kernel.per_token, gemm)
    kernel.process_weights_after_loading(layer)
    assert layer.weight.dtype == current_platform.fp8_dtype()
    assert layer.weight_scale.numel() == n

    x = torch.randn(*x_shape, k, dtype=torch.bfloat16, device="cuda")
    out = kernel.apply_weights(layer, x)

    assert out.shape == (*x_shape, n) and out.dtype == torch.bfloat16
    # per-token FP8 activations are the only lossy step
    assert _rel_err(out, x.float() @ w_ref.t()) < 0.05


@torch.inference_mode()
def test_empty_and_strided_inputs(enable_ptpc):
    n, k = TUNED_SHAPES[1]
    layer = _make_layer(n, k)
    w_ref = dequant_mxfp8_to_bf16(layer.weight.data, layer.weight_scale.data).float()
    kernel = RocmPtpcMxfp8LinearKernel(Mxfp8LinearLayerConfig((n, k)))
    kernel.process_weights_after_loading(layer)

    empty = torch.empty(0, k, dtype=torch.bfloat16, device="cuda")
    assert kernel.apply_weights(layer, empty).shape == (0, n)

    x = torch.randn(100, 3, k, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    out = kernel.apply_weights(layer, x)
    assert out.shape == (3, 100, n)
    assert _rel_err(out, x.float() @ w_ref.t()) < 0.05


def test_non_bf16_model_keeps_emulation(set_flags, monkeypatch: pytest.MonkeyPatch):
    """The per-token kernels compute and return bf16, so an fp16 model stays on
    the emulation."""
    set_flags("1", "hipblaslt")
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float16))
    monkeypatch.setattr(
        "vllm.model_executor.kernels.linear.mxfp8.rocm_ptpc."
        "get_current_vllm_config_or_none",
        lambda: config,
    )
    kernel = init_mxfp8_linear_kernel(weight_shape=TUNED_SHAPES[1])
    assert type(kernel) is EmulationMxfp8LinearKernel


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


@torch.inference_mode()
def test_layerwise_reload_requantizes_in_place(enable_ptpc, dist_init):
    """A weight reload requantizes the new MXFP8 checkpoint tensors into the
    same storage, which captured CUDA graphs point at."""
    n, k = TUNED_SHAPES[1]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ReplicatedLinear(
                k,
                n,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=ModelOptMxFp8Config.from_config({"quant_method": "mxfp8"}),
            )

        def load_weights(self, weights):
            params = dict(self.proj.named_parameters())
            for name, weight in weights:
                params[name].weight_loader(params[name], weight)

    def checkpoint(seed: int) -> list[tuple[str, torch.Tensor]]:
        torch.manual_seed(seed)
        layer = _make_layer(n, k)
        scale = layer.weight_scale.data[:, : k // 32]
        return [("weight", layer.weight.data), ("weight_scale", scale)]

    def reference(weights) -> torch.Tensor:
        return dequant_mxfp8_to_bf16(weights[0][1], weights[1][1]).float()

    with torch.device("cuda"):
        model = Model()
    # as vLLM's loader does, right after the model is built
    record_metadata_for_reloading(model)
    model.load_weights(checkpoint(0))
    model.proj.quant_method.process_weights_after_loading(model.proj)
    assert type(model.proj.quant_method.kernel) is RocmPtpcMxfp8LinearKernel
    ptr = model.proj.weight.data_ptr()

    x = torch.randn(64, k, dtype=torch.bfloat16, device="cuda")
    for seed in (1, 2):
        weights = checkpoint(seed)
        initialize_layerwise_reload(model)
        model.load_weights(weights)
        finalize_layerwise_reload(model, SimpleNamespace(dtype=torch.bfloat16))
        assert model.proj.weight.data_ptr() == ptr
        out = model.proj(x)[0]
        assert _rel_err(out, x.float() @ reference(weights).t()) < 0.05
