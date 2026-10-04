"""Select and validate the compute backend before loading data or opening services."""

import torch


def resolve_device(name: str = "auto", gpu: int = 0) -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU; explicit requests never silently fall back."""
    if name == "auto":
        if torch.cuda.is_available():
            name = f"cuda:{gpu}"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda":
        name = f"cuda:{gpu}"
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("Invalid device. Use auto, cpu, cuda, cuda:N, or mps.") from exc
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("Unsupported device. Use auto, cpu, cuda, cuda:N, or mps.")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Install a CUDA-enabled PyTorch build or use --device cpu.")
        if device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {device.index}.")
    elif device.type == "mps":
        if device.index not in {None, 0} or not torch.backends.mps.is_available():
            raise ValueError(
                "MPS is unavailable. Use a supported Apple GPU and PyTorch build or --device cpu."
            )
        device = torch.device("mps")
    elif device.index is not None:
        raise ValueError("Use --device cpu without an index.")
    return device


def autocast_dtype(device: torch.device) -> torch.dtype:
    """Use float16 on CUDA devices without native bfloat16 support."""
    with torch.cuda.device(device):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
