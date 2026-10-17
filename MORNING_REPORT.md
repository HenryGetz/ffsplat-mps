# MORNING_REPORT — ffsplat on Apple Silicon (MPS)

**Branch:** `apple-silicon` (forked to `https://github.com/HenryGetz/ffsplat`, never pushed upstream, never force-pushed)
**Base commit:** `3fb6444c5faed6de9aa55fa2313df6c2a4104193` (upstream `w-m/ffsplat` HEAD, 2025-07-10)
**Machine:** M1 Max MacBook Pro, 10 CPU cores, 32 GiB unified memory, macOS 27.0.0 (darwin), arm64 — everything below ran locally on this Mac.
**Verdict:** all three stages completed, and the follow-up review (Phase A: parity, Phase B: performance;
`PROGRESS.md` §Phase A / §Phase B) then resolved every remaining profile, corrected two numbers that had been
measured on the wrong code path, and made the MPS full-size encode 1.31x faster. The honest gaps are listed
at the bottom — sections marked **[Phase A/B]** were revised by that review.

Full chronological evidence with verbatim command output: [`PROGRESS.md`](PROGRESS.md).

---

## 1. The gate (F1–F4) — PASSED

| | result |
|---|---|
| **F1 fixtures** | real 3DGS PLY obtained: **INRIA `truck/point_cloud/iteration_30000`** (630,225,580 B, sha256 `65ecf405…`), range-extracted from INRIA's 14.6 GB `models.zip` (the server sends `Accept-Ranges: bytes`, so only the truck members were downloaded). Plus the real **Tanks&Temples COLMAP** set (`tandt_db.zip`, 251 truck images + `sparse/0`), plus HF `dylanebert/3dgs` bonsai as a second real PLY. |
| **F2 machine** | `uname -m` → `arm64`; `torch.backends.mps.is_available()` → `True` (torch 2.14.1, Python 3.13.12). |
| **F3 mpsplat gate** | `pytest` → **`35 passed, 28 skipped in 38.26s`**; real truck PLY rendered through mpsplat's `rasterization(...)` with ffsplat's exact kwargs (`radius_clip=3`, `render_mode="RGB"`, `backgrounds`, `sh_degree=3`, `@torch.no_grad()`) at 978x545 in 5.73 s cold / 1.72 s warm. |
| **F3 render correctness** | **PSNR 20.43 dB** against the real ground-truth frame `000001.jpg`, per-channel correlation 0.91/0.92/0.93, luma correlation 0.915; an independent vision judge says `same_scene_clearly` / `a_3d_object_on_background` (p=1.0). |
| **F4 decision** | **proceed to Stages 1–3** (pytest passes **and** the render visibly depicts the scene). No fallback adapter was needed; no other backend was substituted. |

Artifacts: `gate_mpsplat_render.png`, `gate_mpsplat_render_vs_gt.png`.

---

## 2. Status against the seven done-criteria

| # | criterion | status | evidence |
|---|---|---|---|
| 1 | clean-clone `uv sync`, **no** CUDA packages | **DONE** | `uv pip list` has **0** matches for `cupy|nvidia|cuda`; only `torch`-metadata `Requires-Dist: nvidia-*-cu12 ; platform_system == "Linux"` markers remain, which cannot apply on macOS. Verified again in a *fresh clone* (`ffsplat-clean`) where `uv sync` takes 2.5 s. |
| 2 | convert round-trip cpu **and** mps, float32 | **DONE** | full-size truck: 2,541,226 → 2,540,836 primitives (SOG-web prunes to `floor(sqrt(N))²`), 630,225,580 B PLY → 32,984,494 B (cpu) / 32,994,991 B (mps) container; float32 in/out asserted; bbox relative error 1.7e-05; every one of the 59 PLY properties inside its codec-derived tolerance (**0 failures on both devices**), with numerically identical error statistics across devices. cpu 863.6 s encode / 6.3 s decode; **mps 663.8 s encode / 5.8 s decode**. See PROGRESS.md §1.8 and §Stage 3. |
| 3 | live / view / eval render the real PLY on MPS + PNGs | **DONE** | `view_mps.png`, `live_mps.png` (978x545, mean pixel 0.491, non-black 100%), `view_mps_browser.png` (real `ffsplat view` CLI driven by a browser: 1600x1125, mean 0.609, viewer reports `Connected Rendering`), `eval_mps.png`. **`live` was additionally launched as a real process** (entry point, not an import) and driven by a browser: `live_mps_browser.png` (1600x1125, mean pixel 0.693, viewer reports `Showing scene: 0` / `Rendering`), and its GUI **Convert** button runs to completion on MPS — `live_mps_convert_completed_tiny.png` records `SOG-web` container written + `Showing scene: 1`, while `live_mps_convert.png` / `live_mps_convert_inprogress.png` record the full-size attempt still encoding when the ~30 min budget expired. |
| 4 | full upstream test suite, failures explained | **DONE** | `pytest --doctest-modules` → **13 passed** (2 upstream + 11 added). Zero failures. `mypy` clean, `deptry` clean, `ruff check`/`ruff format --check` clean for every file this branch creates or edits; the 5 remaining ruff findings and 3 format complaints are **pre-existing upstream**, proven by `git diff --quiet 3fb6444` on the files and lines that carry them. |
| 5 | no reachable CUDA runtime path on MPS | **DONE** | `grep -rn cuda src/ffsplat/` → 12 hits, all comments/docstrings except `torch.cuda.is_available()` (the availability predicate itself) and one `torch.cuda.synchronize()` inside an `elif device.type == "cuda"` branch. Proven at runtime: with 14 `torch.cuda.*` entry points trapped to raise, an MPS run of device resolution + PLY decode + `view.render_fn` + `eval.rasterize_splats` reported **0 trapped calls**. |
| 6 | fork README: Apple Silicon section + measured timings | **DONE, revised in Phase B** | `README.md` §"Apple Silicon (MPS) support" now carries a stage-by-stage device policy, a parity-vs-CPU table for all 7 profiles, and re-measured timings: raster **0.09 s cold / 0.033 s warm on the Metal kernels** (the earlier 5.73/1.72 s were the *fallback* path), PLAS 55.7 s cpu vs **30.4 s mps** at 512² and 621.6 s vs **278.7 s** at the real 1594² grid, k-means 105.8→66.4 s cpu / 68.2→**22.3 s** mps, full-size encode **324.8 s mps** vs the 425.8 s baseline and 863.6 s cpu. |
| 7 | small commits, no force-push | **DONE** | 7 commits on `apple-silicon` (§7), pushed to the fork only; never force-pushed, never opened upstream. |

---

## 3. Exact reproduce commands

Everything below is a copy-paste of what was actually run on this Mac.

```bash
# ---- install (Apple Silicon, no CUDA toolkit) -------------------------------
git clone https://github.com/HenryGetz/ffsplat.git && cd ffsplat
git checkout apple-silicon
uv sync

# ---- sanity: MPS present, no CUDA packages ----------------------------------
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available(), hasattr(torch.mps,'compile_shader'))"
uv pip list --python .venv/bin/python | grep -iE "cupy|nvidia|cuda" || echo "NO CUDA PACKAGES"

# ---- convert round-trip on MPS ----------------------------------------------
PLY=/Users/henrygetz/Downloads/ffsplat-fixtures/raw/truck/point_cloud.ply
.venv/bin/ffsplat convert --input $PLY --input-format 3DGS-INRIA.ply \
    --output /tmp/sog --output-format SOG-web --device mps
.venv/bin/ffsplat convert --input /tmp/sog --input-format smurfx \
    --output /tmp/back --output-format 3DGS-INRIA-ply --device mps
# (or: .venv/bin/python /Users/henrygetz/Downloads/ffsplat-fixtures/roundtrip_check.py mps $PLY mps_full_final)

# ---- view (interactive; needs a browser at localhost:8080) ------------------
.venv/bin/ffsplat view --input $PLY --input-format 3DGS-INRIA.ply

# ---- live -------------------------------------------------------------------
.venv/bin/ffsplat live --input $PLY --input-format 3DGS-INRIA.ply

# ---- eval against the real COLMAP set ---------------------------------------
.venv/bin/ffsplat eval --data-path /tmp/sog \
    --dataset-path /Users/henrygetz/Downloads/ffsplat-fixtures/tandt_db/tandt/truck \
    --results-path /tmp/evalout

# ---- tests + gates ----------------------------------------------------------
.venv/bin/python -m pytest --doctest-modules -q     # 13 passed
.venv/bin/ruff check --no-fix . ; .venv/bin/ruff format --check .
.venv/bin/mypy ; .venv/bin/deptry .
uv lock --locked

# ---- proof that no torch.cuda.* call happens on MPS -------------------------
.venv/bin/python /Users/henrygetz/Downloads/ffsplat-fixtures/no_cuda_path_check.py \
    /Users/henrygetz/Downloads/ffsplat-fixtures/raw/truck_tiny202.ply
```

Fixtures are **not** in the repo; they live in `/Users/henrygetz/Downloads/ffsplat-fixtures` (provenance and
sha256 in PROGRESS.md §0.3). The `eval` command needs them.

---

## 4. Bugs found in upstream `ffsplat` (both blocking, both fixed here)

1. **`ffsplat eval` could not start at all.** `cli/eval.py:166` called
   `decode_gaussians(args.data_path, "smurfx")` while `scene_decoder.decode_gaussians` requires `verbose`
   (`TypeError: decode_gaussians() missing 1 required positional argument: 'verbose'`). Present at base
   commit `3fb6444`; fixed by passing `verbose=False`.
2. **`ffsplat convert --output-format 3DGS-INRIA-ply` raised `ValueError: could not broadcast input array
   from shape (N,1) into shape (N,)`.** `scene_decoder._create_scene` deliberately unsqueezes `opacities`
   to `[N,1]` and the `to_field` op preserves that shape, but a PLY vertex property is a scalar.
   Reproduced with an *identity* PLY→PLY conversion on untouched upstream files; fixed at the writer
   boundary in `io/ply.py` (squeeze a trailing singleton, and raise a named error instead of an opaque
   numpy one for genuinely multi-component fields). Regression tests in `tests/test_ply_io.py`.

---

## 5. Known MPS gaps / honesty section

- **`uint16`/`uint32` are not representable on MPS.** The SOG codecs use them only as intermediate
  carriers while splitting a value into bytes, so `ffsplat.device.encoding_dtype` substitutes
  `int32`/`int64` on MPS only. Byte-identical output, but it is a backend limitation.
- **[Phase B] PLAS is NOT slower on MPS.** The earlier claim came from a 2-channel *random* grid; the SOG
  templates feed PLAS a real 6-channel grid, and on that input MPS is **1.8x faster at 512² (30.4 s vs
  55.7 s) and 2.2x faster at the real 1594² grid (278.7 s vs 621.6 s)**. The full-size encode is faster on
  MPS at every stage except the `SOG-PlayCanvas` manhattan k-means (see the new gaps below).
- **`torch` had to be bumped** from the pinned `2.6.0` to `2.14.1`: the vendored MPS rasterizer needs
  `torch.mps.compile_shader`, absent in 2.6.0. Stage-1 convert was validated on 2.6.0 first.
- **`ffsplat live` was launched as a real process and its Convert button was driven through the GUI**; the
  GUI conversion **completes on MPS**, but only on the 40,804-primitive truck scene. On the full-size run
  (2,541,226 primitives, the GUI's default `SOG-PlayCanvas` profile) the click fired and the encode was
  executing on Metal at 94-100% GPU — this port's `KMeans` printed up to `iteration 10/10 done` — but it had
  not finished when the ~30 min best-effort budget expired, so no container was written for it and the
  process was stopped. `live`'s `live preview` checkbox and its Evaluation tab were **not** driven through
  the GUI; the functions behind them are the same `convert`/`eval` code paths verified from the CLI.
- **[Phase A] All 7 profiles are now individually run on both devices** (202² and 512² real-truck scenes),
  decoded back where the format allows, and compared order-independently: all pass their codec-derived
  budgets. Four device-independent template bugs were found and fixed (SOG-web-nosh and
  3DGS-INRIA-nosh-ply split `sh` with `[1]` where `[1, 15]` is required; SOG-web-sh-split used
  `to_fields:` instead of `to_field:`; SOG-PlayCanvas' fixed codebook texture needs >= 65536
  primitives) — each with a regression test in `tests/test_format_profiles.py`.
- **[Phase A] Three further device-independent codec defects are reported, not fixed**: `SOG-web-nosh`
  and `SOG-web-sh-split` write `f_dc` as a 3-pixel-wide AVIF with `chroma: 420` (f_dc error 0.9956 vs
  0.0417/0.0184 for the 444 profiles); a `SOG-web-nosh` container cannot be re-encoded through
  `ffsplat convert` (its `sh` is width 1, every PLY profile splits 16); and SOG-PlayCanvas quaternions are
  lossy for the ~9% of primitives whose `|w|` is not the largest component.
- **[Phase B] The raster timings recorded in Stage 0/3 were measured through mpsplat's torch fallback**
  (`gate_render.py` passed `backgrounds=`), not the Metal kernels: 5.13 s cold / 1.69 s warm is the
  fallback, **0.09 s / 0.033 s is the Metal path** (44.8x). `eval` used that fallback too whenever a
  white background was requested; it now composites afterwards and stays on Metal —
  `tests/test_metal_path.py` fails if any of the three render entry points leaves the Metal kernels.
- **[Phase B] `SOG-PlayCanvas` is the one profile that can be slower on MPS than on CPU** (its `manhattan`
  k-means: 48.0 s vs 38.2 s per assignment pass, because L1 has no matmul identity and MPS reaches ~25% of
  its bandwidth on the blocked subtraction). A fused L1+argmin Metal kernel would be needed.
- **[Phase B] PLAS is 86% of the full-size encode (291.5 s of 339.4 s)** and is at its ceiling for this
  stack: the cost is upstream `plas`'s per-iteration `.item()` sync (5398 of the encode's 5469 host syncs),
  and patching a PyPI wheel would not survive a clean clone.
- **SOG output is not reproducible run-to-run** — upstream, not a port issue: ffsplat calls
  `sort_with_plas` without a `seed`, and PLAS then consumes unseeded `np.random.randint` block shifts and
  `torch.randperm`. Two identical CPU invocations produced different container bytes (PROGRESS.md §1.8).
  Comparisons therefore have to be order-independent, which the validator is.
- **`pre-commit run -a` was not run** — its `ruff --fix`/`prettier` hooks rewrite files (they silently
  rewrote two unrelated upstream files, reverted in `b807068`). The tools `make check` delegates to
  (`ruff --no-fix`, `ruff format --check`, `mypy`, `deptry`) were all run instead and are clean.
- **The eval metrics** come from the container the MPS convert produced: `PSNR 24.295, SSIM 0.8461, LPIPS 0.184`, 0.045 s/image, 2,540,836 Gaussians, ~31.45 MB — i.e. the MPS render agrees with the real photographs (the 3DGS paper reports ~25.2 dB for truck). The metrics are identical after the Phase-B k-means change; the 0.106 s/image figure quoted earlier was taken on a loaded machine and is not attributable to any code change (a COLMAP dataset has `white_background = False`, so `eval` was already on the Metal path).
- `nvidia-ncore`, `imagecodecs` and the `viser` fork's other optional extras were not needed and are not
  installed.

---

## 6. Evidence files at the repo root

| file | what it is |
|---|---|
| `PROGRESS.md` | the full chronological record, verbatim command output at every stage boundary |
| `gate_mpsplat_render.png` | F3 gate render of the real truck PLY via mpsplat `rasterization` on MPS (801,712 B) |
| `gate_mpsplat_render_vs_gt.png` | same, stacked above the real ground-truth photo |
| `view_mps.png` | `ffsplat view`'s own `render_fn` output on MPS, real COLMAP camera (978x545) |
| `view_mps_browser.png` | browser screenshot of the real `ffsplat view` CLI running on MPS |
| `live_mps.png` | `ffsplat live`'s `render_fn` output on MPS |
| `live_mps_browser.png` | browser screenshot of the real `ffsplat live` CLI running on MPS (1600x1125) |
| `live_mps_convert_inprogress.png` | that CLI's GUI **Convert** pressed: `Converting scene...`, button disabled, encoder dir created |
| `live_mps_convert.png` | same, full-size truck, still encoding (`SOG-PlayCanvas`) when the budget expired |
| `live_mps_convert_completed_tiny.png` | GUI **Convert completed** on MPS: `SOG-web` container written, `Showing scene: 1` |
| `eval_mps.png` | `ffsplat eval`'s own `[ground-truth | render]` canvas (`eval_0000.png`) |
| `vendor/mpsplat/VENDOR.md` | provenance of the vendored MPS rasterizer |

## 7. Commits on `apple-silicon`

Phase A/B review commits are listed at the end of `PROGRESS.md`; the list below is the original port.

```
01bb110 Stage 0: PROGRESS.md with F1/F2/F3 evidence + F4 gate decision (MPS gate PASS)
d364612 Stage 1: MPS-capable convert path (drop CUDA deps, pure-torch kmeans, MPS dtype/PLAS-fallback, PLY scalar write)
167d83a Stage 2: vendor mpsplat 1.5.3 (MPS Metal provider), one resolve_device()/synchronize() helper, remove all hard-coded CUDA device sites
b807068 Revert unrelated ruff --fix churn in _renderer.py / get_dynamic_params (not part of this port)
c2ed58d Stage 2/3: fix eval startup (missing verbose), satisfy mypy, exclude vendored upstream tree from ruff/deptry/pre-commit, record gates
36827f4 Stage 3: Apple Silicon README section, MORNING_REPORT, full-size MPS round-trip + eval evidence
# 7th commit: Stage 3 clean-clone validation results, timings note, this list
```
