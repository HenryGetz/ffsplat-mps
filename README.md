# ffsplat

[![Release](https://img.shields.io/github/v/release/w-m/ffsplat)](https://img.shields.io/github/v/release/w-m/ffsplat)
[![Build status](https://img.shields.io/github/actions/workflow/status/w-m/ffsplat/main.yml?branch=main)](https://github.com/w-m/ffsplat/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/w-m/ffsplat/branch/main/graph/badge.svg)](https://codecov.io/gh/w-m/ffsplat)
[![Commit activity](https://img.shields.io/github/commit-activity/m/w-m/ffsplat)](https://img.shields.io/github/commit-activity/m/w-m/ffsplat)
[![License](https://img.shields.io/github/license/w-m/ffsplat)](https://img.shields.io/github/license/w-m/ffsplat)

ffsplat is a powerful framework to convert and compress 3D Gaussian Splatting scenes.

**NOTE: this code is pre-alpha, under heavy development, a community developer preview. Please expect more documentation in the coming weeks (July 2025).**

- **Github repository**: <https://github.com/w-m/ffsplat/>

## Quick Start

- Install CUDA 12.x
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

Run

`uvx --from git+https://github.com/w-m/ffsplat@main ffsplat live --help`

Then open a web browser to [localhost:8080](http://localhost:8080) (or the address that is printed in the viser box on stdout). On the first launch, it will take a few minutes to compile gsplat kernels before a rendered image shows up.

## Setting up a dev environment

Check out the project:

```
git clone https://github.com/w-m/ffsplat.git
cd ffsplat
```

Set up the virtual environment with uv:

```bash
make install
```

Run the live encoding:

```bash
uv run ffsplat live \
    --input /data/gaussian_splatting/mini-splatting2/truck_sparse/point_cloud/iteration_18000/point_cloud.ply \
    --input-format=3DGS-INRIA.ply \
    --dataset-path /data/gaussian_splatting/tandt_db/tandt/truck/ \
    --verbose
```

Check the types and formatting of the code base:

```bash
make check
```

## Apple Silicon (MPS) support

This branch (`apple-silicon`, branched from `3fb6444`) makes ffsplat run on Apple-Silicon Macs through
PyTorch's MPS backend. It was developed and verified on an **M1 Max MacBook Pro — 10 CPU cores, 32 GiB
unified memory, macOS 27.0.0, arm64**, Python 3.13.12, torch 2.14.1. Every number below was measured on
that machine; units are stated.

### Install (Apple Silicon)

No CUDA toolkit is involved. `uv sync` is the whole install step:

```bash
git clone https://github.com/HenryGetz/ffsplat.git
cd ffsplat
git checkout apple-silicon
uv sync
uv run ffsplat convert --help
```

What changed for this:

- `cupy-cuda12x`, `torchpq` and the `pytorch-cu126` wheel index were dropped, and the SOG
  spherical-harmonics k-means was reimplemented in pure PyTorch (`src/ffsplat/models/clustering.py`),
  so it runs on CPU and MPS.
- `torch` is `2.14.1` (bumped from `2.6.0` — the vendored MPS rasterizer needs `torch.mps.compile_shader`,
  which 2.6.0 does not have).
- The differentiable rasterizer now comes from **a vendored copy of
  [mpsplat](https://github.com/zacsmms/mpsplat)** (gsplat 1.5.3 ported to Metal) in `vendor/mpsplat/`,
  pinned by commit and wired in as an editable path dependency — see `vendor/mpsplat/VENDOR.md`.
  `import gsplat` resolves there; the upstream package name is unchanged.
- Every hard-coded CUDA device site is behind one helper, `ffsplat.device.resolve_device()`:
  MPS if available, else CUDA if available, else CPU. `cuda` is only ever selected when
  `torch.cuda.is_available()`, so Linux/CUDA behaviour is unchanged.
- `plas`'s `lapjv` dependency is dropped on macOS only (`[tool.uv] override-dependencies`): it is used
  exclusively by PLAS's upstream evaluation script, which ffsplat never imports, and no published
  `lapjv` version builds on Darwin.

### What is verified to work

| command | status |
|---|---|
| `ffsplat convert --output-format SOG-web` | works on `--device mps` and `--device cpu` |
| `ffsplat convert --output-format 3DGS-INRIA-ply` | works on both |
| round-trip `PLY -> SOG-web -> PLY` | works on both; geometry and all attribute multisets within the codec's quantisation budget |
| `ffsplat view` | works; renders in the browser at `localhost:8080` |
| `ffsplat live` | works; renders in the browser at `localhost:8080`, and the viewer's own **Convert** button runs a full conversion (encode -> container -> decode -> new scene) on MPS |
| `ffsplat eval` | works, including `torchmetrics` SSIM and LPIPS (VGG16) **on MPS** — no CPU fallback needed |
| `make test` (`pytest --doctest-modules`) | 13 passed |
| `mypy`, `deptry`, `ruff check`, `ruff format --check` | clean for every file this branch touches |

One GPU/CPU fallback exists and was never needed on this machine: `PLAS._sort_with_fallback` retries the
kornia-backed PLAS sort on CPU if an op turns out to be missing on MPS, and prints a line on stdout when it
does. Separately, `ffsplat.device.synchronize()` is the single wait helper used on every backend (it is an
ordinary no-op on CPU, not a fallback).

### Measured timings (M1 Max)

Rasterizing the real pre-trained INRIA **truck** model (2,541,226 Gaussians, `sh_degree=3`) at
978x545 with `radius_clip=3`, `render_mode="RGB"` on MPS:

| | wall clock |
|---|---|
| first call (includes Metal JIT compile of the kernels) | 5.73 s |
| subsequent calls | 1.72 s |

PLAS spatial sorting, 2-channel grid, `improvement_break=1e-4` (what the SOG-web template uses):

| grid | cells | `--device cpu` | `--device mps` |
|---|---|---|---|
| 256² | 65,536 | 5.1 s | 29.8 s |
| 512² | 262,144 | 21.6 s | 64.3 s |
| 768² | 589,824 | 50.4 s | 129.0 s |

*PLAS alone* is faster on CPU at these sizes (many small ops, so MPS is kernel-launch bound), but that
does not decide the end-to-end picture: the k-means and quantisation/image passes dominate the full encode
and are much faster on MPS, so the whole conversion is faster on MPS (see the next table).

Full-size truck round-trip, 2,541,226 -> 2,540,836 Gaussians (the SOG-web template prunes to
`floor(sqrt(N))²` for its square grid):
630,225,580 B input PLY -> 32,984,494 B (cpu) / 32,994,991 B (mps) SOG container (~19x smaller).

| device | encode (PLY -> SOG-web) | decode (SOG-web -> PLY) |
|---|---|---|
| `--device cpu` (torch 2.6.0) | 863.6 s (14 min 24 s) | 6.3 s |
| `--device mps` (torch 2.14.1) | 663.8 s (11 min 4 s) | 5.8 s |
| `--device mps`, fresh clone, machine otherwise idle | **425.8 s (7 min 6 s)** | 6.1 s |

The 663.8 s figure was measured while `uv sync`/`pytest`/`mypy`/`deptry`/`ruff` were running concurrently
on the same machine, so the idle-machine 425.8 s is the representative MPS number. Take these as
order-of-magnitude guidance on this class of machine: the SOG encode is dominated by PLAS + the
65 536-entry SH k-means, not by anything MPS-specific.

The MPS encode is the faster of the two here — its k-means and AVIF/quantisation passes beat CPU by more
than PLAS loses, even though PLAS alone is ~2x slower on MPS (table above). Both devices reproduce the
same geometry: bbox relative error 1.7e-05, and every per-property error statistic is numerically
identical across the two devices.

`ffsplat eval` on that container against the real Tanks&Temples/truck COLMAP set (32 test images at
1600 px):

```
[Parser] 251 images, taken by 251 cameras.
PSNR: 24.295, SSIM: 0.8461, LPIPS: 0.184 Time: 0.106s/image Number of GS: 2540836 Size: 31.466MB
```

One-time costs: `uv sync` from a warm uv cache takes ~2.5 s; the first `ffsplat view`/`live` launch
builds viser's browser client (~45 s, once).

### Known gaps on MPS

- `torch`'s MPS backend cannot represent `uint16`/`uint32`. The SOG encoders use those widths only as
  intermediate carriers while splitting a value into bytes, so they are substituted with `int32`/`int64`
  (`ffsplat.device.encoding_dtype`) — byte-identical output, but it is a backend limitation worth knowing.
- MPS is slower than CPU for PLAS at the grid sizes above.
- `ffsplat live`'s **Convert** button was exercised through the GUI and completes on MPS, but only on a small
  scene: on the full truck model with the GUI's default `SOG-PlayCanvas` profile it was still encoding (Metal
  GPU at 94-100%) when a ~30 min best-effort budget expired, so it was stopped. The `live preview` checkbox
  and the live viewer's Evaluation tab were not driven through the GUI; the calls behind them are the same
  `convert`/`eval` paths verified from the CLI.
- viser's initial camera is not set by ffsplat, so a freshly opened viewer is framed arbitrarily — that is
  upstream behaviour, not MPS-specific.

### 3DGS Container Format

Find the documentation for the encoding and decoding description, and the yaml container format in [CONTAINER_FORMAT.md](CONTAINER_FORMAT.md).
