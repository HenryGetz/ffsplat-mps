# Vendored: plas

This directory is a **vendored copy** of [fraunhoferhhi/PLAS](https://github.com/fraunhoferhhi/PLAS),
the Parallel Linear Assignment Sorting implementation used for the SOG templates' spatial sort
(`src/ffsplat/models/transformations.py`, `class PLAS`). It replaces the `plas @ git+…@4f1109c9`
dependency in `pyproject.toml`.

| | |
|---|---|
| Upstream | `https://github.com/fraunhoferhhi/PLAS` |
| Pinned commit | `4f1109c94f29a5a6de62bdf8f20cb3ce1ff0e680` |
| Commit date | 2025-07-28 ("Update README.md") |
| Vendored version | `plas` **0.1** (distribution name `plas`, as upstream) |
| License | Apache-2.0 (`LICENSE`, unchanged from upstream) |
| Local patch | `plas/core.py` only — see *What was changed* below |

## Why this copy exists

Upstream `plas/core.py` evaluates the sort's per-block early-exit condition with a host
synchronisation on **every** reorder:

```python
def l2_dist(a, b):
    return distance_with_batch_dim(a, b).sum().item()   # .item() == host sync
```

`reorder_plas`'s inner loop calls that twice per iteration and branches on the resulting Python
float, so on Apple-Silicon MPS every iteration is a full pipeline drain followed by a burst of
tiny kernels. Measured on the real SOG-web input, that is **5398 of the encode's 5469 host
syncs** and 86% of the full-size encode's wall time (`PROGRESS.md`, B2.5/B4). Upstream's own
source carries the same complaint as a TODO ("perf: distance calculation done for the whole
array after sorting for early breaking").

## What was changed

One file, `plas/core.py`. Nothing else in the tree is touched, so `diff -u` against a fresh
clone at the pinned commit shows exactly these edits (78 changed lines including comments).

1. **`solve_assignments_batch_dim` also returns the winning permutation's cost.** `torch.argmin`
   is replaced by `torch.min`, which returns the values `argmin` discards; their sum is the total
   squared distance of the block *after* the winning permutation is applied, i.e. exactly what
   `l2_dist(params_blocky_flat, target_blocky_flat)` evaluates after the reorder. `argmin` and
   `min` cost the same, so this adds no work. Return arity changed from 1 to 2.

   The returned value is `perms_dist` at the **inverse** row of the winning permutation, not at
   the winning row itself, because `reorder_blocky_shuffled` applies the mapping "destination
   position `j` takes the value from shuffled source `perm[j]`", whose cost is
   `sum_j C[perm[j], j]`. The 24 permutations are closed under inversion, so both rows share the
   same *minimum*, but they are different rows with different costs. Using the winning row
   instead is a ~35% error and was caught by a unit test (`tests/test_plas_vendor.py`).

2. **`get_permutations` also returns that inverse-row table**, so the lookup costs one `gather`
   on a 24-element tensor. Return arity changed from 2 to 3.

3. **`reorder_blocky_shuffled` returns the distance** as a third element. Return arity changed
   from 2 to 3.

4. **`reorder_plas(..., check_every=1)`** — the early-exit test runs once every `check_every`
   reorders. `prev_dist` stays a Python `float` exactly as upstream, so the loop still costs
   exactly one sync per block config for the pre-loop distance; the *only* thing that changes is
   that the per-reorder `l2_dist` call (and its sync) now happens once per `check_every`
   reorders. At `check_every == 1` the code calls upstream's own `l2_dist` — same function, same
   summation order, same value, same number of syncs — so that setting is a bit-for-bit no-op and
   is what `plas_patch_equivalence.py` verifies. At `check_every > 1` it uses the distance the
   solver already produced for the reorder it just applied, which is the same value without a
   second full pass over the grid.

   With `check_every > 1` the test compares the distance now against the distance `check_every`
   reorders ago, i.e. it measures the improvement over the last `check_every` reorders instead of
   over the last one. To keep `improvement_break` meaning the same thing, the batched test uses
   the **geometric mean** per-reorder improvement over the window,
   `1 - (cur / prev) ** (1 / check_every)`, which reduces to upstream's `1 - cur / prev` at
   `check_every == 1`. The inner loop still overshoots its stopping point, but by a fraction of a
   window rather than by a whole window. PLAS is unseeded and not reproducible run-to-run to
   begin with (`np.random.randint`, `torch.randperm`), so this is a heuristic change to the
   iteration count, not a correctness change. The reorders themselves, the permutation solver,
   the low-pass target and the applied permutation are untouched.

   One thing this patch does **not** change, and that is worth recording: upstream's solver
   minimises `sum_c C[c, perm[c]]` but `reorder_blocky_shuffled` then applies the mapping
   "destination `j` takes source `perm[j]`", whose cost is `sum_j C[perm[j], j]` — the cost of
   the *inverse* permutation row. The 24 permutations are closed under inversion, so both rows
   share the same minimum, but they are different rows with different costs, so the applied
   reorder does not generally achieve the minimum the solver found (measured on random blocks:
   applied 54.28 vs solver minimum 34.75). That is upstream behaviour, left exactly as it is.

5. **`sort_with_plas(..., check_every=1)`** — plumbs the value through, documented in the
   docstring. The default is `1`, i.e. upstream behaviour, so the patch is opt-in at the call
   site (`src/ffsplat/models/transformations.py` passes the tuned value).

Everything stays `float32` and device-agnostic: the patched path uses only `torch.min`,
`torch.argsort`, `sum`, `gather` and `.item()`, all of which behave the same on CPU, CUDA and MPS.

## Verifying the copy

```bash
git clone https://github.com/fraunhoferhhi/PLAS /tmp/plas-check
git -C /tmp/plas-check checkout 4f1109c94f29a5a6de62bdf8f20cb3ce1ff0e680
diff -u /tmp/plas-check/plas/core.py vendor/plas/plas/core.py    # the five edits above
diff -r /tmp/plas-check/plas vendor/plas/plas --exclude=core.py  # no differences
diff -u /tmp/plas-check/LICENSE vendor/plas/LICENSE              # no differences
```

`ffsplat-fixtures/plas_patch_equivalence.py` checks the stronger property directly: with
`seed` set and `check_every=1`, the vendored sort returns the *identical* `grid_indices` tensor
and sorted parameters as the unpatched upstream file, on the real captured grids and on both
devices.

## Contents

Copied from the pinned commit, unmodified:

```
plas/__init__.py
plas/vad.py
LICENSE            Apache-2.0
README.md          upstream readme (attribution + algorithm description)
setup.py           build metadata (name=plas, version=0.1, unchanged)
```

Modified: `plas/core.py` (the four edits above).

Deliberately **not** vendored (not needed to `import plas`): `eval/`, `examples/`, `img/`
(PLAS's own evaluation scripts and figures; `lapjv`, which only `eval/flas.py` needs, is still
dropped on macOS by `[tool.uv] override-dependencies` in `pyproject.toml`), `.vscode/`.

## Why the package is still imported as `plas`

Like `vendor/mpsplat`, the copy keeps the upstream *package* name so `import plas` resolves to
this tree. `pyproject.toml` therefore depends on the distribution `plas` and points it at this
directory with a uv *editable* path source, so a clean clone builds and imports the patched
source with no runtime dependence on the live upstream repository.
