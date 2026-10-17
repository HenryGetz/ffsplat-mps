"""End-to-end tests for the output-format templates in ``ffsplat/conf/format``.

The templates are YAML scene graphs, so a shape/field-name typo in one of them is only
detected when that profile is actually run. These tests build a small synthetic 3DGS scene
with the same field layout the 3DGS-INRIA PLY decoder produces (``sh`` of shape ``[N, 16, 3]``,
i.e. degree 3) and push it through every profile whose codebook does not depend on the full
sized scene, then decode the container back and check the round-trip invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ffsplat.coding.scene_decoder import decode_gaussians
from ffsplat.coding.scene_encoder import encode_gaussians
from ffsplat.models.fields import Field
from ffsplat.models.gaussians import Gaussians
from ffsplat.models.operations import Operation

# 1000 primitives: not a perfect square, so the SOG-* profiles exercise the
# `prune_by: opacities` perfect-square pruning step as well.
N = 1000


def _synthetic_scene(n: int = N) -> Gaussians:
    """A float32 3DGS scene with the same shapes the INRIA PLY decoder produces."""
    generator = torch.Generator().manual_seed(0)
    op = Operation(input_fields={}, params={"from": {"file": "synthetic"}})
    means = 2.0 * torch.rand(n, 3, generator=generator) - 1.0
    quaternions = torch.randn(n, 4, generator=generator)
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    scales = torch.rand(n, 3, generator=generator) * 0.1
    opacities = torch.rand(n, 1, generator=generator)
    sh = torch.randn(n, 16, 3, generator=generator)
    return Gaussians(
        means=Field(means, op),
        quaternions=Field(quaternions, op),
        scales=Field(scales, op),
        opacities=Field(opacities, op),
        sh=Field(sh, op),
    )


def _roundtrip(tmp_path: Path, output_format: str) -> Gaussians:
    """Encode with ``output_format``, decode the container back, return the decoded scene."""
    out = tmp_path / output_format
    encode_gaussians(_synthetic_scene(), out, output_format, False)
    return decode_gaussians(out, "smurfx", verbose=False)


def test_sog_web_encodes_and_decodes(tmp_path: Path) -> None:
    """Baseline: the reference SOG profile."""
    decoded = _roundtrip(tmp_path, "SOG-web")
    assert decoded.num_gaussians == 961  # 1000 pruned to the largest perfect square
    assert decoded.sh.data.shape[1] == 16
    assert decoded.means.data.dtype == torch.float32


def test_sog_web_nosh_encodes_and_decodes(tmp_path: Path) -> None:
    """The no-SH profile keeps only f_dc; the `sh` split must still cover all 16 coefficients."""
    out = tmp_path / "sog"
    encode_gaussians(_synthetic_scene(), out, "SOG-web-nosh", False)
    assert sorted(p.name for p in out.iterdir()) == [
        "container_meta.yaml",
        "f_dc.avif",
        "means_bytes_0.avif",
        "means_bytes_1.avif",
        "opacities.avif",
        "quaternions.avif",
        "scales.avif",
    ]
    decoded = decode_gaussians(out, "smurfx", verbose=False)
    assert decoded.num_gaussians == 961
    assert decoded.sh.data.shape == (961, 1, 3)
