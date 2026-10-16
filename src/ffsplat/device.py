"""Backend-capability helpers.

PyTorch's Apple-Silicon (MPS) backend can represent fewer tensor dtypes than the
CPU and CUDA backends. Notably ``uint16`` and ``uint32`` are missing::

    TypeError: Trying to convert UInt16 to the MPS backend but it does not have
    support for that dtype.

The SOG encoders use those widths as *intermediate carriers* while a value is
split into bytes (``SimpleQuantize(dtype="uint16") -> SplitBytes``), and the
decoder rebuilds the value from the byte images with ``Combine(method="bytes")``
into ``int32``/``int64``. Nothing therefore depends on the tensor actually
carrying an ``uint16`` dtype, only on the byte pattern.
"""

from __future__ import annotations

import torch

#: dtypes the MPS backend cannot represent, mapped to an exact substitute.
#: The substitutes have a superset of the source range and the same two's
#: complement byte layout for the non-negative values the callers range-check for.
_MPS_DTYPE_SUBSTITUTES: dict[torch.dtype, torch.dtype] = {
    torch.uint16: torch.int32,
    torch.uint32: torch.int64,
}


def encoding_dtype(device: torch.device, dtype: torch.dtype) -> torch.dtype:
    """Return a dtype equivalent to ``dtype`` that ``device`` can represent.

    On CPU/CUDA this is the identity. On MPS, dtypes the backend lacks are
    replaced by a wider signed type; the substitution is exact for the
    byte-splitting codecs (see the module docstring).
    """
    if device.type == "mps":
        return _MPS_DTYPE_SUBSTITUTES.get(dtype, dtype)
    return dtype
