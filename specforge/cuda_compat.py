"""Escape hatches for CUDA driver / kernel-toolchain mismatches.

flashinfer routes its norm ops through CuTe-DSL kernels that are JIT-compiled to
PTX at launch time. The PTX ISA those kernels emit is newer than what an older
driver branch (e.g. R535 / CUDA 12.2) can JIT, so the launch fails with
``cudaErrorInsufficientDriver`` even though every precompiled cubin in torch and
sgl-kernel loads fine under CUDA minor version compatibility.

Setting ``SPECFORGE_DISABLE_FLASHINFER_NORM=1`` swaps sglang's norm entry points
for eager torch equivalents. They are the reference formulations -- accumulation
happens in fp32 and the result is cast back -- so target hidden states stay
faithful; the cost is the fused kernel, not correctness.
"""

import os

import torch

_ENV_FLAG = "SPECFORGE_DISABLE_FLASHINFER_NORM"

_patched = False


def _normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _scale(weight: torch.Tensor, gemma: bool) -> torch.Tensor:
    # Gemma-style norms fold the residual connection into the weight as (1 + w).
    return weight.float() + 1.0 if gemma else weight.float()


def _make_rmsnorm(gemma: bool):
    def rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None):
        dtype = input.dtype
        result = _normalize(input.float(), eps) * _scale(weight, gemma)
        result = result.to(dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    return rmsnorm


def _make_fused_add_rmsnorm(gemma: bool):
    def fused_add_rmsnorm(input, residual, weight, eps=1e-6, enable_pdl=None):
        dtype = input.dtype
        summed = input.float() + residual.float()
        # flashinfer writes the pre-norm sum back into ``residual`` in the
        # original dtype, then normalizes it into ``input``.
        residual.copy_(summed.to(dtype))
        normed = _normalize(summed, eps) * _scale(weight, gemma)
        input.copy_(normed.to(dtype))

    return fused_add_rmsnorm


_REPLACEMENTS = {
    "rmsnorm": _make_rmsnorm(gemma=False),
    "gemma_rmsnorm": _make_rmsnorm(gemma=True),
    "fused_add_rmsnorm": _make_fused_add_rmsnorm(gemma=False),
    "gemma_fused_add_rmsnorm": _make_fused_add_rmsnorm(gemma=True),
}


def maybe_disable_flashinfer_norms() -> bool:
    """Replace sglang's flashinfer norm entry points with eager torch versions.

    No-op unless ``SPECFORGE_DISABLE_FLASHINFER_NORM`` is set. Returns whether
    the patch was applied.
    """
    global _patched

    if _patched or os.environ.get(_ENV_FLAG, "0") not in ("1", "true", "True"):
        return False

    from sglang.srt.layers import layernorm

    replaced = []
    for name, impl in _REPLACEMENTS.items():
        if hasattr(layernorm, name):
            setattr(layernorm, name, impl)
            replaced.append(name)

    if not replaced:
        raise RuntimeError(
            f"{_ENV_FLAG} is set but sglang.srt.layers.layernorm exposes none of "
            f"{sorted(_REPLACEMENTS)}; the norm dispatch has moved and this patch "
            "needs updating."
        )

    _patched = True
    print(f"[cuda_compat] flashinfer norms replaced with torch: {', '.join(replaced)}")
    return True
