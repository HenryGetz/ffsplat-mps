"""Guard: the vendored PLAS copy and its local patch.

``vendor/plas`` is a copy of fraunhoferhhi/PLAS @4f1109c9 (Apache-2.0) with one patch: the sort's
per-block early-exit condition is evaluated once every ``check_every`` reorders instead of on
every reorder, which removes the per-iteration host sync from the inner loop (see
``vendor/plas/VENDOR.md``). This file pins the parts of that patch that can silently rot:

* the vendored tree is what ``import plas`` resolves to;
* the distance the solver hands back is the real post-reorder total (not the solver's own
  objective, which is a different row of the same table), so the early-exit test compares what
  it compared before;
* ``check_every=1`` calls upstream's own ``l2_dist``, so the compatibility setting is a no-op;
* batching the test really does remove the syncs, and ``check_every=1`` keeps the upstream
  default;
* the sort still returns a permutation of every cell at any ``check_every``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import plas
import plas.core as core
import pytest
import torch

VENDOR = Path(__file__).resolve().parent.parent / "vendor" / "plas"


def _grid(channels: int = 6, size: int = 32, seed: int = 0) -> torch.Tensor:
    return torch.rand(channels, size, size, generator=torch.Generator().manual_seed(seed))


def _sort(grid: torch.Tensor, check_every: int) -> tuple[int, int, torch.Tensor]:
    """Sort ``grid``, returning ``(host syncs, reorders, grid_indices)``.

    ``.item()`` is the only host sync the patched loop makes, and ``reorder_blocky_shuffled``
    is called exactly once per reorder, so counting both attributes the syncs exactly.
    """
    syncs = 0
    reorders = 0
    real_item = torch.Tensor.item
    real_reorder = core.reorder_blocky_shuffled

    def counting_item(self: torch.Tensor) -> float:
        nonlocal syncs
        syncs += 1
        return real_item(self)

    def counting_reorder(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        nonlocal reorders
        reorders += 1
        return real_reorder(*args, **kwargs)

    torch.Tensor.item = counting_item  # type: ignore[method-assign]
    core.reorder_blocky_shuffled = counting_reorder  # type: ignore[assignment]
    try:
        _, indices = core.sort_with_plas(
            grid, improvement_break=1e-4, min_blur_radius=8, seed=3, verbose=False, check_every=check_every
        )
    finally:
        torch.Tensor.item = real_item  # type: ignore[method-assign]
        core.reorder_blocky_shuffled = real_reorder  # type: ignore[assignment]
    return syncs, reorders, indices


def test_import_plas_resolves_to_the_vendored_copy() -> None:
    assert Path(plas.__file__).resolve().is_relative_to(VENDOR)


def test_check_every_defaults_to_upstream_behaviour() -> None:
    assert inspect.signature(core.sort_with_plas).parameters["check_every"].default == 1
    assert inspect.signature(core.reorder_plas).parameters["check_every"].default == 1


def test_solver_distance_is_the_cost_of_the_applied_permutation() -> None:
    """The solver returns the cost of the permutation ``reorder_blocky_shuffled`` applies, not
    the (smaller) minimum it was chosen by: the applied mapping is "destination j takes source
    ``best_idx[j]``", so the residual at ``j`` is ``C[best_idx[j], j]``."""
    generator = torch.Generator().manual_seed(2)
    cand_dists = torch.rand(5, 7, 4, 4, generator=generator)
    perms, perms_one_hot, perms_inv_rows = core.get_permutations(torch.device("cpu"))
    perms_dist = torch.einsum("bsce,pce->bsp", cand_dists, perms_one_hot)
    best_perms_idx, applied_dist = core.solve_assignments_batch_dim(cand_dists, device=torch.device("cpu"))
    best = perms_dist.argmin(dim=-1)
    assert torch.equal(best_perms_idx, perms[best])
    # independent reference: the residual of each destination position, i.e. C[best_idx[j], j]
    reference = (
        cand_dists.gather(2, best_perms_idx.unsqueeze(-1).expand_as(cand_dists)).diagonal(dim1=-2, dim2=-1).sum()
    )
    assert float(applied_dist) == pytest.approx(float(reference), rel=1e-6)
    # and it is the cost of the *inverse* row, which is not the minimum row in general
    assert perms_inv_rows.shape == (perms.shape[0],)
    assert torch.equal(
        perms_inv_rows,
        torch.tensor([int((perms == torch.argsort(perms, dim=1)[p]).all(dim=1).nonzero()[0, 0]) for p in range(24)]),
    )


def test_reorder_distance_is_the_post_reorder_total() -> None:
    """The early-exit test's distance must equal ``l2_dist`` after the reorder, or the break
    threshold would compare something else than it did before the patch."""
    generator = torch.Generator().manual_seed(3)
    block_size = 4
    params = torch.rand(3, 6, block_size * block_size, generator=generator)
    target = torch.rand(3, 6, block_size * block_size, generator=generator)
    indices = torch.arange(block_size * block_size).repeat(3, 1).unsqueeze(1)
    _, _, new_dist = core.reorder_blocky_shuffled(params, indices, target, block_size)
    assert float(new_dist) == pytest.approx(core.l2_dist(params, target), rel=1e-5)


def test_batching_removes_the_per_iteration_sync() -> None:
    grid = _grid()
    syncs_1, reorders_1, _ = _sort(grid, check_every=1)
    syncs_4, reorders_4, _ = _sort(grid, check_every=4)
    # upstream: `l2_dist(...).item()` on every reorder, plus one per block config
    assert syncs_1 >= reorders_1
    # patched: at most half a sync per reorder, i.e. at least 2x fewer syncs per reorder.
    # (The absolute counts are not compared: batching overshoots the stopping point, so a
    # batched sort performs *more* reorders - it is the sync rate that the patch changes.)
    assert syncs_4 * 2 <= reorders_4
    assert syncs_4 / reorders_4 < 0.5 * syncs_1 / reorders_1


@pytest.mark.parametrize("check_every", [1, 4, 16])
def test_sort_still_returns_every_cell_exactly_once(check_every: int) -> None:
    grid = _grid()
    _, indices = core.sort_with_plas(
        grid, improvement_break=1e-4, min_blur_radius=8, seed=5, verbose=False, check_every=check_every
    )
    assert indices.numel() == grid.shape[1] * grid.shape[2]
    assert torch.equal(indices.flatten().sort().values, torch.arange(indices.numel()))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is required")
def test_batching_removes_the_sync_on_mps() -> None:
    grid = _grid().to("mps")
    syncs_1, reorders_1, _ = _sort(grid, check_every=1)
    syncs_4, reorders_4, _ = _sort(grid, check_every=4)
    assert syncs_1 >= reorders_1
    assert syncs_4 * 2 <= reorders_4
    assert syncs_4 / reorders_4 < 0.5 * syncs_1 / reorders_1


def test_sort_on_device_passes_the_module_check_every(monkeypatch: pytest.MonkeyPatch) -> None:
    """ffsplat's PLAS call site must pass ``_PLAS_CHECK_EVERY`` explicitly, not rely on a default."""
    from ffsplat.models import transformations as T

    seen: dict[str, object] = {}

    def fake_sort(grid: torch.Tensor, **kwargs: object) -> tuple[None, torch.Tensor]:
        seen.update(kwargs)
        return None, torch.arange(4)

    monkeypatch.setattr(T, "sort_with_plas", fake_sort)
    T.PLAS._sort_on_device(torch.zeros(6, 4, 4), 1e-4, False)
    assert seen["check_every"] == T._PLAS_CHECK_EVERY
    assert seen["improvement_break"] == 1e-4


def test_shipped_check_every_is_upstream_behaviour() -> None:
    """The sampled test is shipped **off**, and that is a measured decision, not an oversight.

    Enabling it removes 72% of PLAS's host syncs (1.095 -> 0.304 per reorder on the real 512^2
    grid) but costs 7% more reorders at 1594^2, where the sync is only 3.5% of a reorder: PLAS
    goes from 279.3 s to 288.1 s (median of 3, alternating). See PROGRESS.md C4/C5/C6. If you
    change this constant, re-measure at 1594^2 - do not assume the 512^2 result carries.
    """
    from ffsplat.models import transformations as T

    assert T._PLAS_CHECK_EVERY == 1
