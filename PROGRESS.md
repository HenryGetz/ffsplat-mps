# ffsplat → Apple Silicon (MPS) port — PROGRESS

Host: Henry's M1 Max MacBook Pro, macOS (darwin 27.0.0), arm64. Local only.
Agent: ffsplat-apple-mps (sole editing agent).
Repo: `/Users/henrygetz/Downloads/ffsplat` — upstream https://github.com/w-m/ffsplat @ `3fb6444`
Branch: `apple-silicon` → pushed to fork `https://github.com/HenryGetz/ffsplat`
Fixtures: `/Users/henrygetz/Downloads/ffsplat-fixtures` (outside repo).

---

## Stage 0 — Workspace, machine check, fixtures

### 0.1 Clone (2026-10-15 22:47)

```
$ cd /Users/henrygetz/Downloads && git clone https://github.com/w-m/ffsplat.git ffsplat
$ cd ffsplat
$ git log -1 --format='%H %ad %s' --date=short
3fb6444c5faed6de9aa55fa2313df6c2a4104193 2025-07-10 Merge pull request #34 from w-m/package
$ git checkout -b apple-silicon 3fb6444
Switched to a new branch 'apple-silicon'
$ git log -1 --format='%H'
3fb6444c5faed6de9aa55fa2313df6c2a4104193
```

`ffsplat/` did **not** exist beforehand; fresh clone. `git status --short` empty.
Upstream default HEAD **is** 3fb6444, so `apple-silicon` == upstream HEAD at start.

Fork + push (2026-10-15 22:48):

```
$ gh auth status
github.com
  ✓ Logged in to github.com account HenryGetz (keyring)
  - Token scopes: 'gist', 'read:org', 'repo', 'workflow'
$ gh repo fork --remote=false --clone=false     # -> https://github.com/HenryGetz/ffsplat
$ git remote add fork https://github.com/HenryGetz/ffsplat.git
$ git push -u fork apple-silicon
 * [new branch]      apple-silicon -> apple-silicon
branch 'apple-silicon' set up to track 'fork/apple-silicon'.
```

### 0.2 Machine check (F2) — **PASS** (2026-10-15 22:48)

Raw outputs:

```
$ uname -m
arm64

$ python3 -c "import torch; ..."
ModuleNotFoundError: No module named 'torch'      # /opt/homebrew/bin/python3 = Python 3.14.7
```

torch missing for the system interpreter → created a uv venv (Python 3.13) with the macOS MPS torch wheel:

```
$ cd /Users/henrygetz/Downloads && uv venv --python 3.13 mpsenv
Using CPython 3.13.12
Creating virtual environment at: mpsenv
$ uv pip install --python mpsenv/bin/python torch
 + torch==2.14.1
$ mpsenv/bin/python -c "..."
torch 2.14.1
py 3.13.12
mps_available True
mps_built True
cuda_available False
```

**MPS available → proceed.** (`torch 2.14.1` is what the macOS wheel resolves to today; repo pins 2.6.0 — see Stage 1.)

Toolchain: `uv 0.9.7`, `git 2.52.0`, `gh` authed as HenryGetz, 10 CPU cores, 32 GiB RAM, 92 GiB free on `/`.

### 0.3 Fixtures (F1) — **PASS**

Repo README references `.../truck_sparse/point_cloud/iteration_18000/point_cloud.ply` and dataset
`/data/gaussian_splatting/tandt_db/tandt/truck/` (README.md:45-47) — i.e. INRIA truck 3DGS + Tanks&Temples COLMAP.

INRIA release `.../datasets/pretrained/models.zip` is **14,660,630,999 bytes (14.6 GB)** — over budget, but the
server sends `Accept-Ranges: bytes`, so the genuine per-scene members were extracted by HTTP range reads
(`remotezip`), downloading only the truck entries instead of the whole archive.

URL checks (verbatim):

```
--- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/pretrained/truck.zip
HTTP/1.1 404 Not Found
--- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
HTTP/1.1 200 OK
Content-Length: 682628995
--- https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip
HTTP/1.1 200 OK
Accept-Ranges: bytes
Content-Length: 14660630999
$ curl -r 0-15 .../models.zip
HTTP/1.1 206 Partial Content
Content-Range: bytes 0-15/14660630999
00000000: 504b 0304 1400 0000 0000 c7b6 e556 0000  PK...........V..
```

**Primary mandatory fixture — real 3DGS PLY (INRIA truck, iteration_30000)** — extracted by range from `models.zip`:

| item | bytes | sha256 |
|---|---|---|
| `ffsplat-fixtures/raw/truck/point_cloud.ply` | 630225580 | `65ecf4058135a030cddd2198326f67172a4101344b0b54a3fa370cf45ea9688c` |
| `ffsplat-fixtures/raw/truck/cameras.json` | 100406 | `802e9eb9ee7deb034309baecca6d7de2a02b0cb70897a06bdb9ae36ba5bfd5c6` |
| `ffsplat-fixtures/raw/truck/input.ply` | 3673017 | `3c6194b82d1459541477b5fdd887fe5ccaf5af57426116c77172f72a814093fe` |
| `ffsplat-fixtures/raw/truck/cfg_args` | 162 | `ce4b69ffe38781c93921aa590fe5698ee3b531100e9d7522841fa45f7827758e` |

Extraction log: `point_cloud.ply: 630225580 bytes in 56.2s` (range reads only).

**Secondary real 3DGS PLY (HF `dylanebert/3dgs` bonsai, iteration_7000)**:

| item | bytes | sha256 |
|---|---|---|
| `ffsplat-fixtures/raw/bonsai/point_cloud.ply` | 286972500 | `d910c613ffc2410a2e6331757940f5b2f20342ebf7e5203e803e484f75dce4b3` |
| `ffsplat-fixtures/raw/bonsai/cameras.json` | 116695 | `41e623748141d5b1a292c2bcafbf9e897a3876f90c11a14618e9ac6190b05af3` |
| `ffsplat-fixtures/raw/bonsai/input.ply` | 5578785 | `bd75034f1d60b2f9d4c8785263ab4905b6ae597ab76c3ae3423cbcc39c61d27b` |
| `ffsplat-fixtures/raw/bonsai/cfg_args` | 167 | `5a557dd442a4127b7531afe0b6ffee1e3a9aa14e2be667271f73f0b2e9f8bbcc` |

**COLMAP dataset (INRIA Tanks&Temples, for eval)**:

| item | bytes | sha256 |
|---|---|---|
| `ffsplat-fixtures/dl/tandt_db.zip` (kept) | 682628995 | `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1` |

Extracted (zip root is `tandt/` + `db/`, no `tandt_db/` prefix) → moved under `ffsplat-fixtures/tandt_db/`:

```
tandt/truck/images/          251 jpg
tandt/truck/sparse/0/        cameras.bin (64 B)  images.bin (62106537 B)  points3D.bin (15602063 B)  project.ini
tandt/train/  db/drjohnson/  db/playroom/
755M   ffsplat-fixtures/tandt_db
```

Total fixture bytes on disk: 1271M (`raw/truck` 613M + `raw/bonsai` 292M) + 755M COLMAP + 651M zip = ~2.6 GB (< 4 GB budget).

Download URLs (verbatim):
- `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip` (range-extracted: `truck/point_cloud/iteration_30000/point_cloud.ply`, `truck/cameras.json`, `truck/cfg_args`, `truck/input.ply`)
- `https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip`
- `https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/bonsai/point_cloud/iteration_7000/point_cloud.ply` (+ cameras.json, input.ply, cfg_args)

### 0.4 mpsplat gate (F3) — **PASS** (2026-10-15 22:56)

Clone: `https://github.com/zacsmms/mpsplat` @ `94c89496f99d74d81e2a298e1f6159cc3978af34` (2026-05-12)
"Add RETURN_TO.md — paused state of the TSDF-prior experiment". Package `gsplat` **1.5.3**, still imported as `gsplat`.

README-exact install:

```
$ cd mpsplat && uv venv --python 3.13
Using CPython 3.13.12
$ uv pip install --python .venv/bin/python -e .
Installed 18 packages in 240ms
 + jaxtyping==0.3.11 ... + numpy==2.5.4 ... + torch==2.14.1 ... + mpsplat==1.5.3 (from file:///Users/henrygetz/Downloads/mpsplat)
$ .venv/bin/python -c "..."
torch 2.14.1
gsplat 1.5.3
mps True
compile_shader True
```

**pytest (verbatim, tail):**

```
$ cd /Users/henrygetz/Downloads/mpsplat && uv pip install --python .venv/bin/python pytest-env scipy scikit-learn
$ .venv/bin/python -m pytest -q -p no:randomly
sssssssssssssssssssssss...................................ss             [100%]
35 passed, 28 skipped in 38.26s
```

(mpsplat README claims "35 passing / 27 legacy CUDA tests cleanly skipped"; we observe 35 passed / **28** skipped.)

**Gate render of the REAL truck PLY on MPS** — script `ffsplat-fixtures/gate_render.py`, verbatim output:

```
torch 2.14.1 | device mps | cuda_available False
PLY: N=2541226 props=62 loaded in 0.1s
sh shape (2541226, 16, 3) -> sh_degree 3
dtypes: torch.float32 torch.float32 torch.float32 torch.float32 torch.float32
camera img=000001 native=1957x1091 render=978x545 pos=[ 3.3989623  0.683534  -2.2991545 ]
info keys: ['batch_ids', 'camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics', 'opacities',
            'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids', 'flatten_ids', 'isect_offsets',
            'width', 'height', 'tile_size', 'n_batches', 'n_cameras']
render shape (545, 978, 3) dtype torch.float32 device mps:0
first render 5.73s (includes Metal JIT compile)
warm render 1.72s
pixel stats: min 0.021013498 max 1.0 mean 0.5162653
wrote /Users/henrygetz/Downloads/ffsplat/gate_mpsplat_render.png 801712 bytes
```

**How the tensors + camera were built from the PLY** (mirrors ffsplat's own
`conf/decoding_templates/3DGS_INRIA_ply_decoding_template.yaml` exactly):

- read 62 float32 vertex props with numpy: `x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3`
- `means   = stack(x,y,z)`                                   → `[N,3]`
- `quats   = stack(rot_0..rot_3)`                             → `[N,4]` (passed raw; gsplat normalises internally)
- `scales  = exp(stack(scale_0..scale_2))`                     (template `remapping: exp`)
- `opacities = sigmoid(opacity).squeeze()`                     (template `remapping: sigmoid`)
- `colors  = concat( f_dc.reshape(N,1,3), f_rest.reshape(N,3,15).permute(0,2,1) )` → `[N,16,3]`, `sh_degree=3`
- camera: the scene's **real COLMAP camera 0** from `raw/truck/cameras.json`
  (`position`, `rotation`=R camera-to-world, `fx=1163.2547`, `fy=1156.2804`, `1957x1091`), downscaled 2×:
  `c2w = [[R, position],[0,1]]`, `K = [[fx/2,0,W/2],[0,fy/2,H/2],[0,0,1]]`,
  `viewmats = inv(c2w)[None]`, `Ks = K[None]`

Call mirrors ffsplat's own signatures (`cli/view.py:34` and `cli/eval.py:36`), inside `@torch.no_grad()`:

```python
rasterization(means_t, quats_t, scales_t, opac_t, colors,
              viewmat, Ks, W, H,
              sh_degree=3, radius_clip=3, render_mode="RGB", backgrounds=torch.ones(1,3))
```

**Render correctness (independent, quantitative + visual):**

```
size 978x545
MSE=0.009056  PSNR=20.43 dB
gt   mean 0.4944721045957642  render mean 0.5143177585725073
per-channel corr: [0.9101, 0.9187, 0.9339]
luma corr: 0.9154
render frac pixels < 0.95 luma: 0.96158608656498
gt     frac pixels < 0.95 luma: 0.9650419316710753
```

Vision-judge verdict on (GT photo, MPS render):
`{"verdict": "same_scene_clearly" (p=1, conf=1), "depicts": "a_3d_object_on_background" (p=1, conf=1)}`

Artifacts: `gate_mpsplat_render.png` (801712 B), `gate_mpsplat_render_vs_gt.png` (GT top / render bottom).

### 0.5 Gate decision (F4)

| branch | condition | outcome |
|---|---|---|
| ✅ **proceed** | pytest passes **AND** render visibly depicts the scene | **TAKEN** |

- pytest passes: `35 passed, 28 skipped` (the 28 are the suite's own `cuda_only` legacy skips).
- render correct: PSNR 20.43 dB vs the real photograph, luma corr 0.915, judge = same scene, clear 3D object.

**→ Proceed autonomously to Stage 1, then Stage 2, then Stage 3. No permission checkpoints.**

---

## Stage 1 — CONVERT (MPS-capable pure-PyTorch path)

### 1.1 Packaging: drop CUDA from the dependency graph

Edits to `pyproject.toml`:

- removed `"torchpq>=0.3.0.6,<0.4.0"` and `"cupy-cuda12x>=13.4.1,<14.0.0"` (torchpq was imported at
  module scope by `models/transformations.py` and dragged in cupy-cuda12x);
- removed the `[[tool.uv.index]]` block pointing at `https://download.pytorch.org/whl/cu126`;
- `[tool.deptry.per_rule_ignores] DEP002 = ["torchvision"]` (cupy entry gone) — `grep -n "cupy\|cuda\|torchpq" pyproject.toml` → nothing;
- kept `torch==2.6.0` (**tried first as instructed — it resolves and runs on macOS arm64**, no bump needed for Stage 1);
- added `[tool.uv] override-dependencies = ["lapjv; sys_platform != 'darwin'"]` — see 1.2.

### 1.2 Blocker found & worked around: `lapjv` cannot build on Darwin

`uv sync` initially failed:

```
  × Failed to build `lapjv==1.3.27`
  ╰─▶ Call to `setuptools.build_meta.build_wheel` failed (exit status: 1)
        File "<string>", line 53, in <module>
      KeyError: 'Darwin'
  help: `lapjv` (v1.3.27) was included because `ffsplat` (v0.1) depends on `plas` (v0.1) which depends on `lapjv`
```

Evidence it is a dead dependency, not a needed one:

```
$ grep -rn "lapjv" <PLAS @4f1109c>/plas/     -> (none)
$ grep -rn "lapjv" <PLAS @4f1109c>/          -> eval/flas.py:10:import lapjv as lap
$ sed -n '12,20p' <PLAS>/plas/core.py
import numpy as np
import torch
import time
import kornia
...
$ lapjv setup.py, every version 1.3.24/1.3.26/1.3.27/1.3.29:
  # "Darwin": [*UNIX_CXXFLAGS],  not supported anymore due to M1, PRs welcome
```

PLAS's runtime package (`plas/core.py`, `plas/vad.py`) imports only numpy/torch/kornia/tqdm; `lapjv` appears
only in PLAS's upstream evaluation script `eval/flas.py`, which ffsplat never imports. Dropped on macOS only
(the override carries `sys_platform != 'darwin'`, so Linux/CUDA resolution is unchanged). Also verified no
published `lapjv` version ships a macOS wheel (`1.3.22` has only x86_64 cp38/39/310).

`uv sync` then succeeded on Python 3.13.12.

### 1.3 Clean-environment evidence: **no CUDA packages**

```
$ uv pip list --python .venv/bin/python | wc -l
132
$ uv pip list --python .venv/bin/python | grep -icE "^(cupy|nvidia|cuda)"
0
$ uv pip list --python .venv/bin/python | grep -iE "cupy|nvidia|cuda"
(NONE)
$ grep -riE "cupy-cuda|nvidia-cuda|pytorch-cu1" .venv/lib/python3.13/site-packages/*.dist-info/METADATA
  torch-2.6.0.dist-info/METADATA:Requires-Dist: nvidia-cuda-nvrtc-cu12==12.4.127; platform_system == "Linux" and platform_machine == "x86_64"
  (… same for nvidia-cuda-runtime/cupti; all marked Linux+x86_64, i.e. inapplicable on macOS …)

$ uv tree --python .venv/bin/python --depth 1
Resolved 146 packages in 18ms
ffsplat v0.1
├── camorph v0.3.6
├── gsplat v1.4.0
├── jaxtyping v0.2.38
├── jsonargparse v4.40.0
├── numpy v2.2.6
├── opencv-python v4.12.0.88
├── pillow v11.3.0
├── pillow-heif v0.22.0
├── plas v0.1
├── plyfile v1.1.2
├── pyyaml v6.0.2
├── torch v2.6.0
├── torchmetrics v1.7.4
├── torchvision v0.21.0
├── tqdm v4.67.1
└── viser v0.2.23

$ .venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available(), hasattr(torch.mps,'compile_shader'))"
2.6.0 True False
```

**Noted for Stage 2**: `torch.mps.compile_shader` is **absent in torch 2.6.0** (`False`). mpsplat's Metal
kernels require it (`gsplat/mps/__init__.py: is_metal_available()`), so Stage 2 must bump torch. Recorded then.

### 1.4 `transformations.py:17` — torchpq k-means replaced (MPS-capable, pure PyTorch)

- removed the top-level `from torchpq.clustering import KMeans`; `transformations.py` now does
  `from .clustering import KMeans`.
- new module `src/ffsplat/models/clustering.py`: float32-only, device-agnostic k-means supporting
  `euclidean` / `cosine` / `manhattan`, chunked so the `N x K` score matrix is never materialised.
  Because the SOG templates ask for `K = 65536`, centroids are fitted on a bounded subsample
  (`min(N, max(4K, 200_000))`, capped at `max_samples`) followed by one chunked full-data assignment pass,
  so every primitive gets a label consistent with the returned centroids.
- API kept identical to the part of torchpq's surface `Cluster.apply` used: `.fit([D,N]) -> [N] int64`,
  `.centroids -> [D,K] float32`, `KMeans(n_clusters=, distance=, verbose=)`.
- `grep -rn torchpq src/ tests/` → only prose in docstrings.

### 1.5 MPS dtype gap: `uint16` is not representable

`--device mps` failed in `convert_to_dtype`:

```
File "src/ffsplat/models/transformations.py", line 39, in convert_to_dtype
    field_data = field_data.to(torch.uint16)
TypeError: Trying to convert UInt16 to the MPS backend but it does not have support for that dtype.
```

`uint16`/`uint32` are only **intermediate carriers** here: `SplitBytes` immediately widens to
`int32`/`int64` and masks with `0xFF` (line 670), and the decoder rebuilds the value from the byte images
into `int32`/`int64` (`Combine`, line 1011). Nothing depends on the tensor really being `uint16`.

- new module `src/ffsplat/device.py` → `encoding_dtype(device, dtype)` maps `uint16->int32`,
  `uint32->int64` **on MPS only**; identity on CPU/CUDA.
- `convert_to_dtype` casts through it for uint8/uint16/int32/uint32.

### 1.6 Upstream blocker found & fixed: `encode_ply` rejected `[N, 1]` fields

`ffsplat convert ... --output-format 3DGS-INRIA-ply` failed on **pristine upstream code**:

```
$ .venv/bin/ffsplat convert --input truck_tiny202.ply --input-format 3DGS-INRIA.ply \
      --output /tmp/ident.ply --output-format 3DGS-INRIA-ply --device cpu
File "src/ffsplat/io/ply.py", line 28, in encode_ply
    vertex_data[field_name] = field.data.cpu().numpy()
ValueError: could not broadcast input array from shape (40804,1) into shape (40804,)
```

Reproduced with an *identity* conversion, and `git diff --stat HEAD -- src/ffsplat/io/ply.py src/ffsplat/coding/scene_decoder.py`
is empty (both files untouched by this port) ⇒ pre-existing. Cause: `scene_decoder._create_scene` deliberately
does `opacities_field.data.unsqueeze(-1)` → `[N,1]`, and the `to_field` op keeps that shape, but a PLY vertex
property is a scalar. Fixed at the writer boundary: `encode_ply` now squeezes a trailing singleton dim and
raises a named `ValueError` for genuinely multi-component fields (previously an opaque numpy broadcast error).
Regression tests: `tests/test_ply_io.py`.

### 1.7 PLAS (pure PyTorch + kornia) validated on CPU **and** MPS

PLAS runs unmodified (`plas == 0.1` @ `4f1109c`). Device comes from the input tensor. kornia
`gaussian_blur2d` works on both backends:

```
DEVICE: cpu
  kornia gaussian_blur2d OK shape=(1, 3, 256, 256) in 0.003s
  sort_with_plas OK grid=(2,64,64) idx.shape=(1, 64, 64) dev=cpu in 0.90s
DEVICE: mps
  kornia gaussian_blur2d OK shape=(1, 3, 256, 256) in 0.555s
  sort_with_plas OK grid=(2,64,64) idx.shape=(1, 64, 64) dev=mps:0 in 13.49s
```

Added the instructed per-call CPU fallback (`PLAS._sort_with_fallback`): if `sort_with_plas` raises
`RuntimeError`/`NotImplementedError` on a non-CPU device it retries once on CPU, prints
`[PLAS] sort_with_plas failed on device '…' (…); retrying on CPU.` and moves the result back.
**It never triggered on this machine** — no such line appears in any run.

PLAS scaling (2-channel grid, `improvement_break=1e-4`, as the SOG-web template uses):

| grid | cells | cpu | mps |
|---|---|---|---|
| 256² | 65 536 | 5.1 s | 29.8 s |
| 512² | 262 144 | 21.6 s | 64.3 s |
| 768² | 589 824 | 50.4 s | 129.0 s |

(≈linear in cells; the real truck scene needs 1594².)

### 1.8 Round-trip validation

Commands (via `ffsplat-fixtures/roundtrip_check.py <device> <ply> <tag>`):

```
.venv/bin/ffsplat convert --input <ply> --input-format 3DGS-INRIA.ply    --output <w>/sog    --output-format SOG-web        --device <dev>
.venv/bin/ffsplat convert --input <w>/sog --input-format smurfx          --output <w>/out_ply --output-format 3DGS-INRIA-ply --device <dev>
```

Rehearsal on a 202² = 40 804-primitive perfect-square subsample of the real truck PLY
(`ffsplat-fixtures/raw/truck_tiny202.ply`), **both devices PASS**:

```
### device=cpu input=raw/truck_tiny202.ply
encode 18.0s  decode 1.3s
input : N=40804 props=62 format=binary_little_endian bytes=10120922
output: N=40804 props=59 format=binary_little_endian bytes=9631220
sog container bytes=1178832 files=[container_meta.yaml, f_dc.avif, f_rest_centroids.avif,
   f_rest_labels_bytes_0.avif, f_rest_labels_bytes_1.avif, means_bytes_0.avif, means_bytes_1.avif,
   opacities.avif, quaternions.avif, scales.avif]
dropped props: ['nx', 'ny', 'nz']   extra props: []
dtype: input float32 OK / output float32 OK
(nx,ny,nz are the unused 3DGS normals; the 3DGS-INRIA writer does not emit them)
CHECK bbox: max-abs-delta=0.004898 scene-extent=134.885 rel=3.63e-05 (tol abs 0.01)
CHECK opacity(prob): mean_delta=0.000057 median_delta=0.003599 worst-q(0.01,0.5,0.99)_delta=0.003599 (tol 0.05)
CHECK per-property mean/range worst = 0.0160 (tol 0.02)
ROUNDTRIP OK
```

The **MPS** run prints byte-identical CHECK lines (bbox 0.004898 / opacity 0.003599 / mean-range 0.0160) —
`### device=mps`, `encode ~40s`, `ROUNDTRIP OK`. Rationale for the tolerances: the SOG-web template
quantises means to 16 bit, f_dc to 8 bit, rot/scales to 6 bit, opacities to 6 bit (compared in probability
space) and the SH residual to a 5-bit 65 536-entry codebook, so a lossy reconstruction is expected by design.
Gaussian count preserved, float32 in and out, nothing invented.

**Note (upstream, not a port defect): the SOG bytes are not reproducible run-to-run**, because ffsplat calls
`sort_with_plas` without a `seed`, and PLAS then consumes unseeded `np.random.randint` block shifts plus
`torch.randperm`. Evidence — two *identical CPU* invocations of the same command:

```
run1 out sha256: 4c2f3072006905a2c056745fed2340e0884da7329bea61e377e97987f2e41f2d
run2 out sha256: 1f510af3ca94621a488d19022b4da3e244476a810600ae49ca1bc0ccb9fb7022
decoded PLY identical: False
differing sog files between two CPU runs: ['f_dc.avif', 'f_rest_labels_bytes_0.avif',
   'f_rest_labels_bytes_1.avif', 'means_bytes_0.avif', 'means_bytes_1.avif',
   'opacities.avif', 'quaternions.avif', 'scales.avif']
```

`container_meta.yaml` and `f_rest_centroids.avif` (written *before* the sort) are bit-identical every run,
and the decoded value multisets agree to 6 dp — i.e. the primitives are the same set in a different order.
Comparisons therefore have to be order-independent, which is what the checker does.

### 1.9 Tests

`tests/test_clustering.py` (new, 9 tests) + `tests/test_ply_io.py` (new, 2 tests) + upstream
`tests/test_transformations.py`:

```
$ .venv/bin/python -m pytest -q
.............                                                            [100%]
13 passed in 0.92s
```

---

## Stage 2 — LIVE / VIEW / EVAL

### 2.1 Vendored MPS provider

`vendor/mpsplat/` — byte-for-byte copy of `zacsmms/mpsplat` @
`94c89496f99d74d81e2a298e1f6159cc3978af34` (2026-05-12), i.e. **the exact commit whose pytest
(35 passed / 28 skipped) and metal gate render were verified in F3**. 912 KB, all 14
`gsplat/mps/_shaders/*.metal` present, Apache-2.0 `LICENSE` retained.
Provenance + what was/wasn't copied is in `vendor/mpsplat/VENDOR.md`; the hash is also repeated there.

`pyproject.toml`:

- `"gsplat==1.4.0"` → `"mpsplat==1.5.3"` (mpsplat renames only the *distribution*; the top-level
  package is still `gsplat`, so every `from gsplat... import` in ffsplat is unchanged);
- `[tool.uv.sources] mpsplat = { path = "vendor/mpsplat", editable = true }` — editable, because
  upstream's `MANIFEST.in` does not list `gsplat/mps/_shaders/*.metal`, so a built wheel could
  omit the 14 shader sources. No runtime dependence on the live upstream repo.

### 2.2 Torch bump (recorded reason)

`torch==2.6.0` was tried first and **works for Stage 1** (see 1.3). It had to be bumped for Stage 2:

```
$ .venv/bin/python -c "import torch; print(torch.__version__, hasattr(torch.mps,'compile_shader'))"
2.6.0 False
```

The vendored mpsplat's Metal kernels are JIT-compiled by `torch.mps.compile_shader`
(`gsplat/mps/__init__.py::is_metal_available()`), which does not exist in 2.6.0. Bumped to
`torch==2.14.1` — the version mpsplat was tested with in F3 (`True`).

### 2.3 ONE `resolve_device()` helper; all 16 hard-coded CUDA sites removed

`src/ffsplat/device.py` gained:

- `resolve_device(requested=None) -> torch.device` — MPS if available, else CUDA if available, else CPU;
  an explicitly requested but unusable device (the historical hard-coded `"cuda"` / `"cuda:0"` on a Mac)
  warns once and falls back. `cuda` is returned **only** when `torch.cuda.is_available()`, and `mps`
  only when the MPS backend is available ⇒ Linux/CUDA behaviour is unchanged by construction.
- `synchronize(device)` — `torch.mps.synchronize()` / `torch.cuda.synchronize()` / no-op on CPU, so
  every accelerator wait goes through one helper.

Call-site changes (all 16 sites):

| file | was | now |
|---|---|---|
| `cli/convert.py:29` | `--device` default `"cuda:0"` | default `None` → `resolve_device(cfg.device)` |
| `cli/view.py:82` | `.to("cuda")` | `.to(resolve_device())` |
| `cli/view.py:30,31` | `camera.to("cuda")` | `.to(means_t.device)` (device taken from the input) |
| `cli/live.py:275` | `.to("cuda")` | `.to(self.device)` (`self.device = resolve_device()`) |
| `cli/live.py:369` | `.to("cuda")` | `.to(self.device)` |
| `cli/live.py:538,539` | `camera.to("cuda")` | `.to(means_t.device)` |
| `cli/live.py:788` | `device = "cuda"` | `device = self.conv_tool.device` |
| `cli/eval.py:166` | `.to("cuda")` | `.to(resolve_device())` |
| `cli/eval.py:125` | `device = "cuda"` | `device = resolve_device()` |
| `cli/eval.py:35` | `torch.ones(..., device="cuda")` | `device=colors.device` |
| `cli/eval.py:72,73,75,87` | `torch.cuda.Event` + `torch.cuda.synchronize()` | `time.perf_counter()` + `synchronize(device)` |

```python
$ .venv/bin/python -c "from ffsplat.device import resolve_device; print(resolve_device(), resolve_device('cuda:0'), resolve_device('cpu'))"
[device] 'cuda:0' is not usable on this machine; using 'mps' instead.
mps mps cpu
```

### 2.4 `grep -rn "cuda" src/ffsplat/` — every hit classified

```
$ grep -rn "cuda" src/ffsplat/ --include=*.py
src/ffsplat/device.py:45:    if torch.cuda.is_available():                      [guard: the availability check itself]
src/ffsplat/device.py:46:        return torch.device("cuda")                    [guard: unreachable unless is_available()]
src/ffsplat/device.py:55,56,59: docstring                                      [comment/doc]
src/ffsplat/device.py:68:  (device.type == "cuda" and torch.cuda.is_available())  [guard]
src/ffsplat/device.py:87:  docstring                                        [comment/doc]
src/ffsplat/device.py:91:    elif device.type == "cuda":                        [guard: else-branch of the MPS test]
src/ffsplat/device.py:92:        torch.cuda.synchronize()                       [guarded: only in the cuda branch]
src/ffsplat/models/clustering.py:5: docstring                                  [comment/doc]
src/ffsplat/cli/convert.py:34: help text                                      [comment/doc]
src/ffsplat/cli/eval.py:74: comment                                           [comment/doc]
```

Non-comment/non-docstring executable CUDA references, exhaustively: `device.py:45` and `device.py:68`
(both `torch.cuda.is_available()`, i.e. the availability predicate) and `device.py:92`
(`torch.cuda.synchronize()`, inside `elif device.type == "cuda"`). **Claim: there is no reachable
runtime CUDA path on MPS** — verified at the end of Stage 3 with a runtime assertion.

### 2.5 Upstream blocker found & fixed: `ffsplat eval` crashed at startup

`ffsplat eval` could not start on **unmodified upstream code** — `decode_gaussians` requires `verbose`:

```
$ git show 3fb6444:src/ffsplat/cli/eval.py | grep -n decode_gaussians
17:from ..coding.scene_decoder import decode_gaussians
166:    gaussians = decode_gaussians(args.data_path, "smurfx").to("cuda")
$ git show 3fb6444:src/ffsplat/coding/scene_decoder.py | grep -n "def decode_gaussians"
103:def decode_gaussians(input_path: Path, input_format: str, verbose: bool) -> Gaussians:
$

$ .venv/bin/ffsplat eval --data-path ... --dataset-path ...
  File "src/ffsplat/cli/eval.py", line 166, in main
    gaussians = decode_gaussians(args.data_path, "smurfx").to(resolve_device())
TypeError: decode_gaussians() missing 1 required positional argument: 'verbose'
```

Fixed at the call site (`verbose=False`); `live.py`'s two call sites already passed it.

### 2.6 Environment actually validated (`uv sync` on the final pyproject)

```
$ .venv/bin/python -c "import torch, gsplat; ..."
torch 2.14.1 mps True compile_shader True
gsplat -> /Users/henrygetz/Downloads/ffsplat/vendor/mpsplat/gsplat/__init__.py
metal True
$ uv pip list --python .venv/bin/python | grep -iE "cupy|nvidia|cuda"
NO CUDA PACKAGES
$ uv pip list | grep -E "^(torch|torchvision|mpsplat) "
mpsplat    1.5.3    /Users/henrygetz/Downloads/ffsplat/vendor/mpsplat
torch      2.14.1
torchvision 0.29.1
```

The vendored provider is picked up exactly as intended: `import gsplat` resolves inside `vendor/mpsplat`,
`is_metal_available()` is `True`, and there are no CUDA packages.

### 2.7 `ffsplat view` on MPS

(a) The real CLI, driven by a browser:

```
$ .venv/bin/ffsplat view --input .../raw/truck/point_cloud.ply --input-format 3DGS-INRIA.ply
(viser) No client build found. Building now... yarn install ... vite build ... ✓ built in 13.15s
╭──────────────── viser ────────────────╮
│   HTTP      │ http://localhost:8080   │
│   Websocket │ ws://localhost:8080     │
╰───────────────────────────────────────╯
Viewer running... Ctrl+C to exit.
```

Browser (managed Chromium, WebGL2 available) → `http://127.0.0.1:8080`, page reports
`Connected Rendering Max Img Res 64 2048 Render Format:` (i.e. the ffsplat viewer connected and the
renderer is live), 2 canvases (1280x900, 1600x1125). Screenshot saved as
`view_mps_browser.png` (1600x1125, mean pixel **0.6088**, non-black fraction **0.9998**).
Vision judge: `{"browser_ui": "viewer_ui_with_render" (p=1)}` — i.e. a rendered scene inside the
viewer UI. (The render is framed by viser's *default* camera, which the CLI does not set, so the
framing is arbitrary; the judge rates the content `render_present_but_unclear`.)

(b) The render itself, through the CLI's own `render_fn` (the function viser calls every frame) with a
real COLMAP camera from the scene — `ffsplat-fixtures/render_cli_path.py view ...`:

```
mode=view device=mps cuda_available=False
loaded: Gaussians(N=2,541,226, sh_degree=3, device=mps:0)
render_fn returned: <class 'numpy.ndarray'> (545, 978, 3) float32
pixel stats: min=0.0137 max=1.0000 mean=0.4910 nonblack_frac=1.0000
wrote /Users/henrygetz/Downloads/ffsplat/view_mps.png (794384 bytes)
```

### 2.8 `ffsplat live` on MPS

`live.py`'s `InteractiveConversionTool.render_fn` never touches `self`, so it is invoked unbound with the
real PLY's tensors (`ffsplat-fixtures/render_cli_path.py live ...`) — same code path the live viewer uses:

```
mode=live device=mps cuda_available=False
loaded: Gaussians(N=2,541,226, sh_degree=3, device=mps:0)
render_fn returned: <class 'numpy.ndarray'> (545, 978, 3) float32
pixel stats: min=0.0137 max=1.0000 mean=0.4910 nonblack_frac=1.0000
wrote /Users/henrygetz/Downloads/ffsplat/live_mps.png (794384 bytes)
```

`live_mps.png` is byte-identical to `view_mps.png` — expected and honest: `view.py:render_fn` and
`live.py:render_fn` are the same rasterization call with the same tensors and the same camera, so they
produce the same image. `view` is additionally verified through its real CLI + browser above.

### 2.9 `ffsplat eval` on MPS

```
$ .venv/bin/ffsplat eval --data-path <sog container> \
      --dataset-path /Users/henrygetz/Downloads/ffsplat-fixtures/tandt_db/tandt/truck \
      --results-path <out>
Warning: The DataParser uses Camorph to load the data. ...
[Parser] 251 images, taken by 251 cameras.
Running evaluation...
Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to ~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
100%|██████████| 32/32 [00:17<00:00,  2.52it/s]
100%|██████████| 32/32 [00:22<00:00,  1.42it/s]
PSNR: 6.002, SSIM: 0.0826, LPIPS: 0.664 Time: 0.088s/image Number of GS: 40804 Size: 1.123MB
```

- 32 test images (the parser's `test_every=8` split of the real Tanks&Temples/truck COLMAP set).
- **torchmetrics LPIPS (VGG16) and SSIM ran on MPS** — the allowed CPU fallback was never needed.
- Timing now comes from `time.perf_counter()` + `synchronize(device)`.
- `eval_0000.png` (1 551 366 B) is the eval loop's own `[render | ground-truth]` canvas; vision judge on
  it: `{"layout": "two_panels" (p=1), "render_side": "structure_visible" (p=1)}`.
- PSNR is low because this smoke run used the deliberately tiny 40 804-primitive subsample container
  (most of the frame is genuinely empty there), not the full scene.

### 2.10 Repo quality gates (final environment)

```
--- ruff check . ---
Found 5 errors.        [*] 2 fixable with the --fix option
--- ruff format --check . ---
Would reformat: src/ffsplat/cli/eval.py
Would reformat: src/ffsplat/cli/live.py
Would reformat: src/ffsplat/models/transformations.py
3 files would be reformatted, 28 files already formatted
--- mypy ---
Success: no issues found in 11 source files
--- deptry . ---
Success! No dependency issues found.
--- pytest --doctest-modules -q (make test) ---
.............                                                            [100%]
13 passed
--- uv lock --locked ---
Resolved 150 packages
```

All 5 ruff findings and all 3 format complaints are **pre-existing upstream**, not produced by this port:

```
$ for f in src/ffsplat/cli/__main__.py src/ffsplat/datasets/colmapparser.py src/ffsplat/render/_renderer.py; do
      git diff --quiet 3fb6444 -- "$f" && echo "$f UNCHANGED by this port"; done
src/ffsplat/cli/__main__.py                   UNCHANGED by this port
src/ffsplat/datasets/colmapparser.py          UNCHANGED by this port
src/ffsplat/render/_renderer.py               UNCHANGED by this port
$ git diff -U0 3fb6444 -- src/ffsplat/models/transformations.py | grep -c dict.fromkeys
0
```

i.e. 4 of the 5 findings are in files this port never touched, and the 5th (`transformations.py:1527` C420)
is not in the diff. The 3 format complaints are all `f"...{d["key"]}..."` nested-quote style that the
installed ruff 0.12.2 wants normalised; those hunks are upstream text (e.g. `get_table_row`,
`evaluation`'s print) and were deliberately **not** reformatted to avoid unrelated churn. Every file this
port created or edited reports 0 ruff findings and is format-clean:

```
$ for f in device.py models/clustering.py cli/convert.py cli/view.py cli/live.py cli/eval.py io/ply.py; do
      ruff check --no-fix src/ffsplat/$f | grep -c ": [A-Z]"; done   -> 0 for every file
$ ruff format --check src/ffsplat/models/clustering.py src/ffsplat/device.py src/ffsplat/io/ply.py \
      src/ffsplat/cli/convert.py src/ffsplat/cli/view.py tests/
1 file would be reformatted, 8 files already formatted   # only transformations.py (upstream hunks)
```

Two config additions were needed because `vendor/` is third-party code that must stay byte-identical
(`vendor/mpsplat/VENDOR.md`): `[tool.ruff] extend-exclude = ["vendor"]`,
`[tool.deptry] extend_exclude = ["vendor"]` (`extend_` so deptry's own defaults such as `.venv` survive —
plain `exclude` replaced them and produced 41 355 bogus findings), and `exclude: ^vendor/` on the
ruff/ruff-format/prettier pre-commit hooks.

`pre-commit run -a` itself was **not** run: its `ruff --fix`/`prettier` hooks rewrite files (they had already
silently rewritten `_renderer.py` and `get_dynamic_params`, which I reverted in commit `b807068`), and
there is no value in reformatting unrelated upstream code tonight. `ruff check --no-fix`,
`ruff format --check`, `mypy` and `deptry` — the tools `make check` delegates to — were all run as above.

### 2.11 `ffsplat eval` on the full scene, from the container the MPS convert produced

```
$ .venv/bin/ffsplat eval --data-path /tmp/rt_mps_full_final/sog \
      --dataset-path /Users/henrygetz/Downloads/ffsplat-fixtures/tandt_db/tandt/truck \
      --results-path /tmp/eval_full
[Parser] 251 images, taken by 251 cameras.
Running evaluation...
PSNR: 24.295, SSIM: 0.8461, LPIPS: 0.184 Time: 0.106s/image Number of GS: 2540836 Size: 31.466MB

real	0m33.110s
```

That is the expected ballpark for the truck scene (the 3DGS paper reports ~25.2 dB), so the MPS render —
including camorph's camera convention — genuinely reproduces the trained model. `eval_mps.png` is the
eval loop's own `[ground-truth | render]` canvas from this run.

---

## Stage 3 — full-size MPS round-trip (final environment)

```
### device=mps input=raw/truck/point_cloud.ply
encode 663.8s  decode 5.8s
input : N=2541226 props=62 format=binary_little_endian bytes=630225580
output: N=2540836 props=59 format=binary_little_endian bytes=599638774
sog container bytes=32994991 files=[container_meta.yaml, f_dc.avif, f_rest_centroids.avif,
   f_rest_labels_bytes_0.avif, f_rest_labels_bytes_1.avif, means_bytes_0.avif, means_bytes_1.avif,
   opacities.avif, quaternions.avif, scales.avif]
dropped props: ['nx', 'ny', 'nz']   extra props: []
grid prune: N=2541226 -> floor(sqrt(N))^2=2540836 (dropped 390, SOG-web `prune_by: opacities`; output N=2540836)
CHECK dtype: input float32 OK / output float32 OK
  x/y/z      worst max-delta=0.004839 (tol 0.006186, 0.78x budget)
  f_dc_*     worst max-delta=0.082205 (tol 0.193544, 0.42x budget)
  rot_*      worst max-delta=0.032825 (tol 0.196958, 0.17x budget)
  scale_*    worst max-delta=0.214432 (tol 1.286793, 0.17x budget)
  opacity    worst max-delta=0.039440 (tol 0.047619, 0.83x budget)
  f_rest_*(codebook) worst mean/range=0.0127 (tol <= 0.05)
properties checked: 59; tolerance failures: 0
CHECK bbox: max-abs-delta=0.002308 scene-extent=135.135 rel=1.71e-05 (tol 0.01)
CHECK opacity(prob): mean_delta=0.000100 worst-q(0.01,0.5,0.99)_delta=0.002472 (tol 0.05)
 "encode_s": 663.77, "decode_s": 5.84, "sog_bytes": 32994991
ROUNDTRIP OK (mps)
```

**Every CHECK statistic is numerically identical to the CPU run** (x 0.002864, y 0.001131, z 0.004839,
opacity 0.039440, scale_0 0.269486, f_dc_2 0.082205, f_rest_44 mean/range 0.008381, bbox rel 1.71e-05) —
the MPS path is numerically equivalent to CPU, and the only difference is the container size
(32,994,991 B vs 32,984,494 B) from PLAS's unseeded shuffling.

**MPS is faster than CPU for the full encode**: 663.8 s vs 863.6 s.

### 3.1 Runtime proof: no reachable CUDA path on MPS

```
$ .venv/bin/python /Users/henrygetz/Downloads/ffsplat-fixtures/no_cuda_path_check.py raw/truck_tiny202.ply
resolve_device() -> mps | cuda.is_available() -> False
decoded: 40804 gaussians on mps:0
view.render_fn -> (256, 256, 3) mean 0.01976010389626026
eval.rasterize_splats -> (1, 256, 256, 3) in 0.451s
NO REACHABLE CUDA PATH ON MPS: OK (0 trapped torch.cuda calls)
```

(14 public `torch.cuda.*` entry points were replaced with traps that record and raise before the run.)

### 3.2 Clean-clone validation (fresh clone, fresh venv, documented steps only)

A brand-new clone of the fork (`/Users/henrygetz/Downloads/ffsplat-clean`, 0 dirty files) with a fresh
`uv sync`, running only the commands documented in the README. Script:
`ffsplat-fixtures/clean_clone_validation.sh`.

```
===== clean clone: c2ed58dd1eb5561cf3521e1d6c01007634b63bdc ; worktree: 0 dirty files
===== env
python 3.13.12
torch 2.14.1 | mps True | compile_shader True
gsplat -> /Users/henrygetz/Downloads/ffsplat-clean/vendor/mpsplat/gsplat/__init__.py
metal True
===== CUDA packages?
NO CUDA PACKAGES
===== convert PLY -> SOG-web (MPS)
k-means: fitting on a 262144/2541226 subsample (65536 clusters, euclidean)
... iteration 10/10 done
k-means: done - 64161 non-empty clusters of 65536
Successfully converted .../raw/truck/point_cloud.ply to /tmp/cc_sog

real	7m5.804s
===== convert SOG-web -> PLY (MPS)
Successfully converted /tmp/cc_sog to /tmp/cc_back

real	0m6.084s
===== 1 render through the clean clone's own ffsplat.cli.view.render_fn (MPS)
mode=view device=mps cuda_available=False
loaded: Gaussians(N=2,541,226, sh_degree=3, device=mps:0)
render_fn returned: <class 'numpy.ndarray'> (545, 978, 3) float32
pixel stats: min=0.0137 max=1.0000 mean=0.4910 nonblack_frac=1.0000
wrote /tmp/cc_view.png (794384 bytes)
===== eval on the clean clone's own container (MPS)
[Parser] 251 images, taken by 251 cameras.
PSNR: 24.295, SSIM: 0.8461, LPIPS: 0.184 Time: 0.046s/image Number of GS: 2540836 Size: 31.444MB
===== artifacts
-rw-r--r--@ 1 henrygetz  wheel  599638774 Oct 15 23:47 /tmp/cc_back/point_cloud.ply
-rw-r--r--@ 1 henrygetz  wheel    1762941 Oct 15 23:47 /tmp/cc_eval/eval_0000.png
-rw-r--r--@ 1 henrygetz  wheel     794384 Oct 15 23:47 /tmp/cc_view.png
 31M	/tmp/cc_sog
```

- `uv sync` in the fresh clone: 2.5 s (warm uv cache).
- Out PLY is byte-length-identical to the main-repo run (599,638,774 B) and the visualisation PNG is
  byte-length-identical (794,384 B).
- **The eval metrics are identical to the main-repo run** (PSNR 24.295, SSIM 0.8461, LPIPS 0.184) even
  though PLAS shuffled differently, i.e. the port is reproducible where the codec is.
- The MPS encode took **425.8 s** here with the machine otherwise idle, vs 663.8 s in §Stage 3, which was
  measured while `uv sync`/`pytest`/`mypy`/`deptry`/`ruff` were running concurrently. Both are real
  measurements; the idle-machine number is the representative one.

### 3.3 Upstream test suite and gates — results table

| check | command | result | failures tied to MPS? |
|---|---|---|---|
| unit tests | `pytest --doctest-modules -q` (`make test`) | **13 passed** in ~1 s | none — 0 failures |
| lock consistency | `uv lock --locked` | `Resolved 150 packages` | n/a |
| types | `mypy` | **Success: no issues found in 11 source files** | none |
| dependencies | `deptry .` | **Success! No dependency issues found** | none |
| lint | `ruff check --no-fix .` | 5 findings, **all pre-existing upstream** (4 in files this port never touched, the 5th not in the diff) | no |
| format | `ruff format --check .` | 3 files "would reformat", **all pre-existing** upstream nested-quote f-string style | no |

The suite is small (the repo ships 2 test functions); 11 tests were added for the new code
(`tests/test_clustering.py`, `tests/test_ply_io.py`). The port introduces **zero** test failures.

### 3.4 `grep -rn "cuda" src/ffsplat/` — final classification

See §2.4 for the full 12-hit listing. Summary: 7 of the 12 hits are comments/docstrings, 4 are the
availability guard itself (`torch.cuda.is_available()`), and exactly one is an actionable call
(`torch.cuda.synchronize()` inside `elif device.type == "cuda"`, therefore unreachable on MPS).
Runtime confirmation (no trapping, real MPS run) is in §3.1.

---

## Stage 2 follow-up — `ffsplat live` as a REAL process (closing the last gap)

The director flagged that `ffsplat live` had only been exercised through its `render_fn`, never launched as
a process. Closed here.

### 2.12 Launching the real entry point

```
$ .venv/bin/ffsplat live --input /Users/henrygetz/Downloads/ffsplat-fixtures/raw/truck/point_cloud.ply \
      --input-format 3DGS-INRIA.ply --results-path /tmp/live_results
(viser) No client build found. Building now...
 * Install prebuilt node (20.4.0) ... done.
yarn install v1.22.22 ... ✨  Done in 9.07s.
vite v5.4.11 building for production... ✓ built in 10.05s
╭──────────────── viser ────────────────╮
│             ╷                         │
│   HTTP      │ http://localhost:8080   │
│   Websocket │ ws://localhost:8080     │
│             ╵                         │
╰───────────────────────────────────────╯
Loading scene 0...
Viewer running... Ctrl+C to exit.
```

(The `opening handshake failed` / `websockets.exceptions.InvalidMessage: did not receive a valid HTTP
request` traceback that follows is a transient TCP health probe hitting the websocket port, identical to
what `ffsplat view` produced; the viewer then runs normally. It is also what the viewer's own "Stdout Logs"
panel mirrors, because `live.py` wraps `sys.stdout`/`sys.stderr` with `_GuiStdout`, which still forwards to
the real streams.)

### 2.13 Browser drives the real live viewer

Managed Chromium at `http://127.0.0.1:8080`, WebGL2 available. The page reports
`Connected Showing scene: 0 Rendering Max Img Res 64 2048 Render Format:` and exposes the full Convert GUI:

```
buttons: ["Export Canvas", "Reset View", "Scenes", "Convert", "Load scene", "Save scene",
          "Save encoding parameters", "Convert"]
labels:  ["Server", "Max Img Res", "Render Format:", "store at:", "Output format", "live preview",
          "method", "method", "method", "method", "scaling_fn", "shuffle", "improvement_break",
          "means", "sh0", "shN", "opacities", "scales", "quaternions", "n bits" x5,
          "#clusters", "distance", "n bits", "method", "shN_labels", "shN_centroids", "n bits",
          "image_codec", "exact", "lossless", "quality", ...]
```

Screenshot: `live_mps_browser.png` (1600x1125, mean pixel 0.6930, non-black 0.9935).

### 2.14 The GUI's own "Convert" button, pressed on MPS

The Scenes tab was active first, so the Convert tab was clicked to reveal its button (initially
`w=0,h=0,visible=false`, then `w=615` at x=1158,y=537), which was then clicked with a real trusted event.

The viewer's own "Stdout Logs" mirror confirms the GUI callback fired:

```
$ (viewer stdout mirror)  ... websockets.exceptions.InvalidMessage: did not receive a valid HTTP request
                          Converting scene...
```

`Converting scene...` is the first line of `Runner._convert`. Corroborating state, all read from the live
GUI: the Convert button flipped to `disabled: true` (`disable_convert_ui()`), and the encoder's output
directory appeared:

```
$ echo $TMPDIR
/var/folders/3z/2289_9cn5yz7_cc_ql4t80ph0000gn/T/
$ ls -la $TMPDIR/tmpf_7z9elq
drwx------  gaussians1          # live.py's class-level tempfile.TemporaryDirectory()
```

`ps -o time` reports ~0 CPU seconds for the process, which initially looked like a hang; `sample 23208 6`
resolved it — 3464 of 3487 samples are on thread `DispatchQueue_74: metal gpu stream`, with
`com.apple.MPS.MPSLibrary.preheat` and `MTLCompilerConnectionQueue` active, i.e. the GUI conversion is
genuinely executing on the Metal/MPS path (and JIT-preheating MPS kernels), rather than blocking. Wall
clock and CPU-time accounting simply diverge because the work is inside GPU waits.

GPU utilisation while that conversion ran (independent of `ps`, which cannot see GPU-bound time):

```
$ ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*'
"Device Utilization %"=94
```

The encode writes nothing until its trailing `write_file` ops, so an empty `gaussians1/` plus ~94% GPU
utilisation is exactly the shape of a normal in-progress full-size SOG encode (the equivalent CLI run took
425-664 s of encode). In-progress screenshot: `live_mps_convert_inprogress.png`.

### 2.15 Progress inside the GUI conversion (read from the viewer's own stdout mirror)

The GUI conversion is the full-size SOG encode, so it runs for minutes; the viewer's "Stdout Logs" panel is
the only view of it (we did not pass `--verbose`, so there are no per-operation lines, but the k-means
reports itself):

```
$ (viewer stdout mirror, ~20 min in)
k-means: iteration 6/10 done  k-means: iteration 7/10 done  k-means: iteration 8/10 done
k-means: iteration 9/10 done  k-means: iteration 10/10 done
```

That is `ffsplat.models.clustering.KMeans` — the module written for this port — executing inside the GUI
callback on MPS, i.e. the GUI conversion path genuinely runs the ported code.


### 2.16 The GUI conversion path, run to COMPLETION on MPS

Step 2.14 above (full-size truck, SOG-PlayCanvas default profile) was still encoding when the ~30 min
best-effort budget ran out, so the same GUI path was then driven to completion on the small real-truck scene
(`ffsplat-fixtures/raw/truck_tiny202.ply`, 202² = 40,804 primitives, a random subsample of the same
pre-trained INRIA truck model; chosen only because a full-size SOG-PlayCanvas encode through the GUI did not
finish in budget):

```
$ .venv/bin/ffsplat live --input .../raw/truck_tiny202.ply --input-format 3DGS-INRIA.ply --results-path /tmp/live_results
Loading scene 0...
Viewer running... Ctrl+C to exit.
```

The browser first switched the "Output format" dropdown from its default `SOG-PlayCanvas` to the validated
`SOG-web` (verified by reading the Mantine Select's input value back: `before: "SOG-PlayCanvas"` →
`after: "SOG-web"`), then pressed the Convert button. The viewer's own stdout mirror:

```
$ (viewer stdout mirror)
k-means: iteration 8/10 done  k-means: iteration 9/10 done  k-means: iteration 10/10 done
k-means: done - 39693 non-empty clusters of 40804
Loading scene 1...
```

and the header changed from `Showing scene: 0` to `Showing scene: 1`, with the Convert button re-enabled
(`disabled: false`) — i.e. `Runner.run_conversion` finished, `add_scene` added the converted scene and
`load_scene` decoded it again and re-rendered. The encoder's output was written:

```
$ ls -la $TMPDIR/tmpbuqnc62e/gaussians1
container_meta.yaml   4903 B
f_dc.avif            94340 B
f_rest_centroids.avif  553341 B
f_rest_labels_bytes_0.avif  59361 B
f_rest_labels_bytes_1.avif  59510 B
means_bytes_0.avif   132553 B
means_bytes_1.avif    69790 B
opacities.avif        32602 B
quaternions.avif      96336 B
scales.avif           74607 B
1.2M total
$ head -6 container_meta.yaml
container_identifier: smurfx
container_version: '0.1'
packer: ffsplat-0.1
profile: SOG-web
profile_version: 0.1
```

That is the same file set the CLI produces for `SOG-web` (`/tmp/rt_cpu_tiny/sog`), so the GUI's Convert
button runs the identical, verified encoder — including this port's pure-PyTorch `KMeans` — on MPS.

### 2.17 Honest summary of what was and was not driven through the GUI

| GUI element | driven? | outcome |
|---|---|---|
| `ffsplat live` as a real process | yes | started, served `localhost:8080`, `Loading scene 0...` |
| viewer renders the real PLY on MPS | yes | `Connected` / `Rendering`, screenshot `live_mps_browser.png` |
| Convert tab + Convert button, full-size truck, default `SOG-PlayCanvas` | yes (clicked with a real trusted event) | callback fired (`Converting scene...`), encoder dir created, 94-100% GPU on `metal gpu stream`, this port's k-means printed up to `iteration 10/10 done`; **not completed within the ~30 min budget** (SOG-PlayCanvas on 2.54M primitives), so **no** container was written and the process was stopped |
| Convert button, 40,804-primitive truck scene, `SOG-web` | yes | **completed**: container written, scene decoded, `Showing scene: 1` |
| `live preview` checkbox | no | not driven |
| live viewer's Evaluation tab | no | not driven (the eval CLI path is verified separately in §2.9/§2.11) |

Artifacts, with pixel statistics and an independent vision-judge verdict on each (so what each depicts is
recorded explicitly):

| artifact | size | mean pixel | non-black | judge |
|---|---|---|---|---|
| `live_mps_browser.png` | 1600x1125 | 0.6930 | 0.9935 | `rendered_3d_scene` (p=1.0), `yes_controls_visible` (p=1.0) |
| `live_mps_convert_inprogress.png` | 1600x1125 | 0.6989 | 0.9928 | viewer + disabled Convert visible |
| `live_mps_convert.png` | 1600x1125 | 0.6998 | 0.9931 | same, encoding still in progress |
| `live_mps_convert_completed_tiny.png` | 1600x1125 | 0.3955 | 0.6764 | `rendered_3d_scene` (p=1.0), `yes_controls_visible` (p=1.0) |

The live server processes were stopped afterwards; port 8080 is free and no `ffsplat live` process remains.

---

# Phase A — PARITY ("does it fully work the same?")

Harnesses added for this phase (outside the repo, in `ffsplat-fixtures/`):

| file | purpose |
|---|---|
| `profile_matrix.py` | A1: encode+decode one output profile on one device, record container bytes/file hashes and the order-independent error profile |
| `matrix_report.py` | A1: cpu-vs-mps parity table, incl. a **cpu-vs-cpu(2nd run) noise floor** |
| `make_subsample.py` | real-truck subsample fixtures (`raw/truck/point_cloud_512.ply` = 262 144, `_1024.ply` = 1 048 576 primitives) |
| `rasterizer_reference_check.py` | A3: independent pure-PyTorch CPU rasterizer reference |
| `stage_profile.py` | B1: in-process per-transformation stage timing + `ioreg` GPU-utilisation sampling |

Order-independence is mandatory and was honoured everywhere: PLAS is called without a seed upstream, so
two runs of the *same* command on the *same* device differ (PROGRESS §1.8). Every value comparison is a
1001-quantile profile, never element-wise.

## A1. Per-profile matrix, both devices, on the REAL fixtures

### A1.1 The profiles

`src/ffsplat/conf/format/` contains exactly 7 output profiles:

| profile | file | container | structure |
|---|---|---|---|
| SOG-web | `SOG-web.yaml` | avif | means 16-bit, f_dc 8-bit, f_rest = 65536-entry euclidean codebook (16-bit labels), rot/scale/opacity 6-bit |
| SOG-web-png | `SOG-web-png.yaml` | png | as SOG-web, PNG images |
| SOG-web-nosh | `SOG-web-nosh.yaml` | avif | as SOG-web but **shN dropped** (DC only) |
| SOG-web-sh-split | `SOG-web-sh-split.yaml` | avif | f_rest quantised 5-bit and split into 15 per-coefficient images, no codebook |
| SOG-PlayCanvas | `SOG-PlayCanvas.yaml` | webp | means 16-bit, scales uint8, quats unit-sphere+8-bit, sh0+opacity fused into one 8-bit image, shN = 65536-entry **manhattan** codebook packed into a 1024x960 RGB texture |
| 3DGS-INRIA-ply | `3DGS-INRIA-ply.yaml` | ply | float32, lossless structural round trip |
| 3DGS-INRIA-nosh-ply | `3DGS-INRIA-nosh-ply.yaml` | ply | float32, f_rest dropped |

### A1.2 The five real failures found by running them (all device-independent)

Running every profile on the real 202² truck scene (`raw/truck_tiny202.ply`, 40 804 primitives)
immediately surfaced five device-independent defects — four of them template bugs, all reproduced
*identically* on `--device cpu` and `--device mps`:

```
SOG-web-nosh (cpu AND mps: encode):
    File "src/ffsplat/models/transformations.py", line 232, in apply
      chunks = field_data.split(split_size_or_sections, dim)
  RuntimeError: split_with_sizes expects split_sizes to sum exactly to 16 (input tensor's size at
  dimension 1), but got split_sizes=[1]

3DGS-INRIA-nosh-ply (cpu AND mps: encode):
  RuntimeError: split_with_sizes expects split_sizes to sum exactly to 16 (input tensor's size at
  dimension 1), but got split_sizes=[1]

SOG-web-sh-split (cpu AND mps: encode):
    File "src/ffsplat/models/transformations.py", line 743, in apply
      to_field_name = params["to_field"]
  KeyError: 'to_field'

SOG-PlayCanvas (cpu AND mps, tiny scene only: encode, after 169.5 s cpu / 135.6 s mps of work):
    File "src/ffsplat/models/transformations.py", line 571, in apply
      field_data = field_data.reshape(*shape)
  RuntimeError: shape '[-1, 960, 3]' is invalid for input of size 1836180
```

(plus a fifth, found while decoding: a `SOG-web-nosh` container cannot be re-encoded by the shipped PLY
profiles — see A1.5.)

All four are fixed in four separate commits (`b960016`, `d6c34c0`, `3d7f72a`, `d15812b`), each with a
regression test in `tests/test_format_profiles.py` that drives the real template machinery with a
synthetic 3DGS scene in the INRIA PLY field layout (`sh` of shape `[N,16,3]`, degree 3) — no fixture
outside the repo is needed, so the tests run in CI:

```
$ .venv/bin/python -m pytest tests/test_format_profiles.py -q --no-header
6 passed, 73 warnings in 3.89s
```

### A1.3 The parity table (real 202² truck scene, 40 804 primitives, both devices)

`profile_matrix.py` encodes each profile with `--device <dev>`, records the container bytes/file hashes,
decodes back to a PLY, and `matrix_report.py` compares — order-independently — (A) decoded(cpu) vs
decoded(mps), (B) each decoded scene vs the input, and (C) **decoded(cpu, 2nd run) vs decoded(cpu, 1st
run)** as the run-to-run noise floor. Budgets are derived from each template's declared quantiser
(bit depth per property group; for the PlayCanvas opacity/sh0 image from the channelwise min/max recorded
in `container_meta.yaml`). `PASS(rerun noise)` means two runs of the *same device* differ by as much as
cpu-vs-mps — i.e. the difference is the codec's own non-determinism, not the port.

```
$ .venv/bin/python matrix_report.py tiny_fix tiny_fix2      # verbatim (ffsplat-fixtures/)
== 3DGS-INRIA-nosh-ply: cpu=OK mps=OK
   bytes cpu=2286928 mps=2286928 (delta +0)  N=40804->40804  enc cpu=1.1s mps=1.15s
   group       cpu|delta|max  mps|delta|max       budget  CPU vs MPS max  CPU2 vs CPU max  verdict
   f_dc_            0.000000       0.000000     0.000081        0.000000         0.000000  PASS
   opacity          0.000001       0.000001     0.000010        0.000000         0.000000  PASS
   rot_             0.000000       0.000000     0.000030        0.000000         0.000000  PASS
   scale_           0.000000       0.000000     0.000209        0.000000         0.000000  PASS
   x                0.000000       0.000000          n/a        0.000000         0.000000  PASS
   y                0.000000       0.000000          n/a        0.000000         0.000000  PASS
   z                0.000000       0.000000          n/a        0.000000         0.000000  PASS
   device error statistics agree (within 2x) for: ['f_dc_', 'opacity', 'rot_', 'x', 'y', 'z']

== 3DGS-INRIA-ply: cpu=OK mps=OK
   bytes cpu=9633137 mps=9633137 (delta +0)  N=40804->40804  enc cpu=1.11s mps=1.2s
   group       cpu|delta|max  mps|delta|max       budget  CPU vs MPS max  CPU2 vs CPU max  verdict
   f_dc_            0.000000       0.000000     0.000081        0.000000         0.000000  PASS
   f_rest_          0.000000       0.000000     0.000008        0.000000         0.000000  PASS(codebook)
   opacity          0.000001       0.000001     0.000010        0.000000         0.000000  PASS
   rot_             0.000000       0.000000     0.000030        0.000000         0.000000  PASS
   scale_           0.000000       0.000000     0.000209        0.000000         0.000000  PASS
   x                0.000000       0.000000          n/a        0.000000         0.000000  PASS
   y                0.000000       0.000000          n/a        0.000000         0.000000  PASS
   z                0.000000       0.000000          n/a        0.000000         0.000000  PASS
   device error statistics agree (within 2x) for: ['f_dc_', 'f_rest_', 'opacity', 'rot_', 'x', 'y', 'z']

== SOG-web: cpu=OK mps=OK
   bytes cpu=1178135 mps=1177421 (delta -714)  N=40804->40804  enc cpu=21.08s mps=17.1s
   group       cpu|delta|max  mps|delta|max       budget  CPU vs MPS max  CPU2 vs CPU max  verdict
   f_dc_            0.041730       0.041730     0.095852        0.000000         0.000000  PASS
   f_rest_          0.226940       0.226940     codebook        0.054536         0.000000  PASS(codebook)
   opacity          0.040699       0.040699     0.047571        0.000000         0.000000  PASS
   rot_             0.029818       0.029818     0.142421        0.000000         0.000000  PASS
   scale_           0.254005       0.254005     0.996627        0.000000         0.000000  PASS
   x                0.004898       0.004898          n/a        0.000000         0.000000  PASS
   y                0.001000       0.001000          n/a        0.000000         0.000000  PASS
   z                0.003895       0.003895          n/a        0.000000         0.000000  PASS
   device error statistics agree (within 2x) for: ['f_dc_', 'f_rest_', 'opacity', 'rot_', 'scale_', 'x', 'y', 'z']

== SOG-web-nosh: cpu=OK mps=OK
   bytes cpu=498971 mps=499030 (delta +59)  N=40804->40804  enc cpu=12.13s mps=10.36s
   group       cpu|delta|max  mps|delta|max       budget  CPU vs MPS max  CPU2 vs CPU max  verdict
   f_dc_            0.995600       0.995600     0.095100        0.083263         0.036874  PASS
   opacity          0.040699       0.040699     0.047571        0.000000         0.000000  PASS
   rot_             0.029818       0.029818     0.142421        0.000000         0.000000  PASS
   scale_           0.254005       0.254005     0.996627        0.000000         0.000000  PASS
   x                0.004898       0.004898          n/a        0.000000         0.000000  PASS
   y                0.001000       0.001000          n/a        0.000000         0.000000  PASS
   z                0.003895       0.003895          n/a        0.000000         0.000000  PASS
   device error statistics agree (within 2x) for: ['f_dc_', 'opacity', 'rot_', 'scale_', 'x', 'y', 'z']

== SOG-web-png: cpu=OK mps=OK
   bytes cpu=1308890 mps=1307869 (delta -1021)  N=40804->40804  enc cpu=24.08s mps=16.56s
   group       cpu|delta|max  mps|delta|max       budget  CPU vs MPS max  CPU2 vs CPU max  verdict
   f_dc_            0.018411       0.018411     0.095852        0.000000         0.000000  PASS
   f_rest_          0.226940       0.226940     codebook        0.054536         0.000000  PASS(codebook)
   opacity          0.040699       0.040699     0.047571        0.000000         0.000000  PASS
   rot_             0.029818       0.029818     0.142421        0.000000         0.000000  PASS
   scale_           0.254005       0.254005     0.996627        0.000000         0.000000  PASS
   x                0.004898       0.004898          n/a        0.000000         0.000000  PASS
   y                0.001000       0.001000          n/a        0.000000         0.000000  PASS
   z                0.003895       0.003895          n/a        0.000000         0.000000  PASS
   device error statistics agree (within 2x) for: ['f_dc_', 'f_rest_', 'opacity', 'rot_', 'scale_', 'x', 'y', 'z']

== SOG-web-sh-split: cpu=OK mps=OK
   bytes cpu=1033144 mps=1032754 (delta -390)  N=40804->40804  enc cpu=13.37s mps=10.75s
   group       cpu|delta|max  mps|delta|max       budget  CPU vs MPS max  CPU2 vs CPU max  verdict
   f_dc_            0.995600       0.848104     0.095100        0.147496         0.147496  PASS(rerun noise)
   f_rest_          0.089436       0.089436     0.078928        0.000000         0.000000  PASS(codebook)
   opacity          0.040699       0.040699     0.047571        0.000000         0.000000  PASS
   rot_             0.029818       0.029818     0.142421        0.000000         0.000000  PASS
   scale_           0.254005       0.254005     0.996627        0.000000         0.000000  PASS
   x                0.004898       0.004898          n/a        0.000000         0.000000  PASS
   y                0.001000       0.001000          n/a        0.000000         0.000000  PASS
   z                0.003895       0.003895          n/a        0.000000         0.000000  PASS
   device error statistics agree (within 2x) for: ['f_dc_', 'f_rest_', 'opacity', 'rot_', 'scale_', 'x', 'y', 'z']

# summary
3DGS-INRIA-nosh-ply    cpu=2286928 mps=2286928 parity=PASS
3DGS-INRIA-ply         cpu=9633137 mps=9633137 parity=PASS
SOG-web                cpu=1178135 mps=1177421 parity=PASS
SOG-web-nosh           cpu=498971 mps=499030 parity=PASS
SOG-web-png            cpu=1308890 mps=1307869 parity=PASS
SOG-web-sh-split       cpu=1033144 mps=1032754 parity=PASS
```

Reading of the table:

* **means / rot / scale / opacity are bit-identical across devices** for every SOG profile (cpu-vs-mps
  delta 0.000000 at the 6-decimal print precision) — the whole quantise → byte-split → image path is
  device-independent.
* `f_rest` differs only inside the k-means codebook (SOG-web/-png: 0.0545 max, 0.00022 mean/range against
  a 0.05 mean/range budget; the same numbers on cpu and mps when compared to the input) — expected: an
  argmin tie in the Lloyd iterations can flip a label, and the two devices' matmul accumulation order
  differs. It is far inside the codec budget.
* The only >budget cpu-vs-mps number in the whole matrix is `SOG-web-sh-split`'s `f_dc_` (0.147496), and
  the **cpu-vs-cpu rerun gives exactly the same 0.147496** — that profile writes f_dc as a 3-px-wide
  AVIF with `chroma: 420`, so 4:2:0 chroma subsampling reorders the loss, and PLAS's unseeded shuffling
  changes that loss run to run. Not a device effect.

### A1.4 Codec defects found by the matrix (device-independent, reported not silently changed)

1. **`SOG-web-nosh` and `SOG-web-sh-split` destroy `f_dc` with 4:2:0 chroma subsampling.** Both write
   f_dc with `image_codec: avif, quality: 100, chroma: 420` from a 3-pixel-wide image (`[N,3]`), i.e. the
   chroma planes are subsampled below one sample per primitive. Measured: `f_dc` error vs the input is
   **0.9956** for SOG-web-nosh and **0.9956** for SOG-web-sh-split, versus **0.0417** for SOG-web and
   **0.0184** for SOG-web-png, which use `chroma: 444` / `quality: -1`. In the 3DGS RGB mapping
   (`rgb = 0.5 + 0.282 * f_dc`) that is a ~0.28 full-scale colour error, on both devices identically.
   **Not changed**: `chroma` is a declared parameter of the external SOG layout, so switching it is a
   format-compatibility decision, not a port fix; the numbers are recorded here instead.
2. **`SOG-PlayCanvas` needs the full 65536-entry codebook.** Its reshape is a fixed 1024x960 RGB texture
   (64 clusters x 45 coefficients per row), while `KMeans` clamps `n_clusters` to the sample count, so
   any scene with fewer than 65536 primitives cannot be packed into it. Verified identical failure on
   `--device cpu` (169.5 s) and `--device mps` (135.6 s) on the 40 804-primitive scene. The template now
   documents the constraint and `Reshape` raises a named `ValueError`
   (`Cannot reshape field 'shN_centroids' of shape (40804, 45) (1836180 elements) to [-1, 960, 3]`)
   instead of a bare `RuntimeError`; `tests/test_format_profiles.py` pins it. The 512² (262 144-primitive)
   matrix below exercises this profile successfully on both devices.
3. **A `SOG-web-nosh` container is not re-encodable through `ffsplat convert`.** It decodes to an `sh` of
   width 1, and every shipped PLY profile splits a *16-wide* `sh`:
   `RuntimeError: split_with_sizes expects split_sizes to sum exactly to 1 (input tensor's size at
   dimension 1), but got split_sizes=[1, 15]` — identical on cpu and mps. The A1 harness decodes such a
   container in-process and zero-pads the DC-only `sh` back to 16 before writing the PLY (the nosh PLY
   profile discards the padding); the gap itself is left as reported.

## A2. The unverified surface: mpsplat's 28 skipped tests

`vendor/mpsplat` is the byte-identical mpsplat provider. Its suite on this machine:

```
$ cd /Users/henrygetz/Downloads/mpsplat && .venv/bin/python -m pytest -v -p no:randomly --no-header
======================= 35 passed, 28 skipped in 26.39s ========================
```

**Every one of the 35 passing tests lives in the four MPS-specific files** (nothing else runs on a Mac,
because there is no CUDA extension and no CUDA device):

| passing file | tests | what it covers on MPS |
|---|---|---|
| `tests/test_mps_metal.py` | 23 | the native Metal kernels: projection (EWA + UT), tile intersection, `rasterize_to_pixels_3dgs_fwd/_bwd`, spherical harmonics, 2DGS kernels |
| `tests/test_mps_autograd.py` | 6 | the Metal kernels' autograd wiring (`rasterize_to_pixels_3dgs_bwd`, SH backward) |
| `tests/test_mps_smoke.py` | 3 | end-to-end smoke renders |
| `tests/test_mps_cpu_parity.py` | 3 | MPS vs CPU numeric parity of the Metal path |

The 28 skips, with the reason verbatim from pytest and whether ffsplat can reach that capability:

| # | test(s) | reason (verbatim) | capability | does ffsplat use it? |
|---|---|---|---|---|
| 12 | `test_2dgs.py::test_projection_2dgs[batch_dims0/1/2]`, `::test_fully_fused_projection_packed_2dgs[batch_dims0/1/2-False]`, `::test_rasterize_to_pixels_2dgs[batch_dims0/1/2-3/31]` (3+3+6) | `2DGS support wasn't built` | 2D Gaussian splatting | **no** — ffsplat renders 3DGS only (`scene.primitives: 3DGS-INRIA`) |
| 9 | `test_math.py::test_quat_to_rotmat`, `::test_rotmat_to_quat`, `::test_rotmat_to_quat_roundtrip`, `::test_quat_multiply`, `::test_quat_inverse`, `::test_quat_rotate`, `::test_quat_slerp`, `::test_safe_normalize`, `::test_quat_normalize_rotation` | `legacy CUDA-only test, skipped on device=mps` | the CUDA quaternion/math kernels | **partly** — ffsplat never calls these kernels; the equivalent maths is exercised *through* the MPS tests of the projection/rasterizer and through the A3 reference below |
| 2 | `test_strategy.py::test_strategy`, `::test_strategy_requires_grad` | `3DGS support isn't built in` | densification/pruning strategies (training) | **no** — see the forward-only proof |
| 1 | `test_compression.py::test_png_compression` | `legacy CUDA-only test, skipped on device=mps` | the CUDA png/compression kernel | **no** — ffsplat's image writing is `cv2`/`Pillow` + `pillow_heif` on the host, never mpsplat's codec |
| 1 | `test_ftheta.py::test_rasterization[RGB]` | `3DGUT support isn't built in` | UT / ftheta camera models | **no** — ffsplat is pinhole-only (`--input-format 3DGS-INRIA.ply`, `Ks` is a plain pinhole matrix) |
| 1 | `test_basic.py` (module-level) | `gsplat CUDA extension not available` | the CUDA `_C` extension build | no — that *is* the point of the port |
| 2 | `test_cameras.py` (module-level) | `CUDA required for camera model tests` | `create_camera_model` / rolling shutter | **no** — ffsplat builds `viewmats`/`Ks` itself from camorph/COLMAP |

**ffsplat is forward-only and never trains.** Verbatim:

```
$ grep -rn --include=*.py -E "\.backward\(\)|requires_grad|torch\.optim|Optimizer|\.train\(\)|autograd\." src/ffsplat/
(exit 1: no matches)

$ grep -rn --include=*.py "no_grad" src/ffsplat/
src/ffsplat/models/clustering.py:132:        with torch.no_grad():
src/ffsplat/cli/view.py:17:@torch.no_grad()
src/ffsplat/cli/live.py:525:    @torch.no_grad()
src/ffsplat/cli/eval.py:26:@torch.no_grad()

$ grep -rn --include=*.py "rasterization(" src/ffsplat/
src/ffsplat/cli/view.py:37: ... rasterization(
src/ffsplat/cli/live.py:546: ... rasterization(
src/ffsplat/cli/eval.py:38: ... rasterization(
```

Consequence: the 28 skips remove **no capability ffsplat needs**. The only caveat is that the CUDA-path
tests being skipped means mpsplat's Metal path is validated by its own MPS test files, not by the CUDA
reference tests — which is exactly why A3 below adds an *independent* reference.

## A3. Independent rasterizer reference (not mpsplat's own suite)

`ffsplat-fixtures/rasterizer_reference_check.py` implements the 3DGS renderer from the paper in plain
PyTorch on the CPU and shares no code with mpsplat:

* EWA projection: `means_c = R mu + t`; `J = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]`;
  `Sigma = R S S^T R^T`; `Sigma_2d = J W Sigma W^T J^T + 0.3 I`; conic `[S22, -(S01+S10)/2, S11]/det`.
* Depth-sorted front-to-back alpha compositing over **all** pixels and **all** gaussians
  (`alpha = clamp(opacity * exp(-(0.5 (a dx^2 + c dy^2) + b dx dy)), max=0.99)`, sample point
  `(col+0.5, row+0.5)`), degree-0 SH colour `0.5 + 0.28209 * f_dc`.
* The conventions were matched to the published ones (citations in the script header:
  `vendor/mpsplat/gsplat/cuda/_torch_impl.py:240-311`, `_constants.py:18`), and this was verified by
  comparing the reference's own `means2d` / `depths` / `conics` with mpsplat's `fully_fused_projection`
  for the same input on both devices — they agree to the printed precision:
  `mine conics [[0.2355518937110901, 0.17499999701976776, 0.6168176531791687], ...]` vs
  `mps conics [[0.23555190861225128, 0.17499999701976776, 0.6168174743652344], ...]`.
* Scene: 300 real gaussians selected from the real INRIA truck PLY (the most opaque ones whose 3.33-sigma
  footprint lies fully inside the frame, 0.5-12 px sigma), real COLMAP camera 0 at 1/8 resolution
  (244x136), `sh_degree=0`, `radius_clip=0`, `backgrounds=None`.

Verbatim result:

```
$ .venv/bin/python rasterizer_reference_check.py
# rasterizer reference check
PLY .../raw/truck_tiny202.ply N=40804  camera 000001 1957x1091 -> 244x136 (1/8)
candidates fully inside frame with 0.5-12 px sigma: 18131; selected 300 by descending opacity
reference: mean=0.016917 max=1.947017 covered(alpha>1/255)=0.1581
mps : shape=(136, 244, 3) mean=0.016956  MAE(all)=4.191e-05  MAE(covered,5247px)=2.560e-04  max=8.193e-02  p99.9=1.089e-02  px(|d|>0.01)=118/33184  PSNR=59.96dB  corr=0.999935
cpu : shape=(136, 244, 3) mean=0.016956  MAE(all)=4.192e-05  MAE(covered,5247px)=2.561e-04  max=8.193e-02  p99.9=1.089e-02  px(|d|>0.01)=118/33184  PSNR=59.96dB  corr=0.999935

RESULT: reference vs mpsplat on ['mps', 'cpu'] -> AGREE (MAE over the image < 5e-3 and >=99% of pixels within 0.01)
```

Tolerance logic and its limits, stated explicitly:

* the criterion is **MAE over the whole image < 5e-3** (measured 4.2e-05, i.e. 100x inside) **and at least
  99% of pixels within 0.01** (measured 99.64%): the residual is ≤ 0.4% of pixels, worst pixel 0.082 on a
  1.95-peak image.
* the tail is **not** reproduced by either of gsplat's own cut-offs applied alone to the reference: adding
  the `alpha < 1/255` threshold gives MAE 1.05e-04 (worse), and adding a 3.33-sigma per-pixel footprint cut
  gives MAE 1.69e-02 (much worse, because a diagonal-based radius is not the ellipse's own axes). So the
  tail is attributed to the rasterizer's tile-local culling/early-out interacting with strongly anisotropic
  footprints, not to a device or projection error — the projected means, depths **and conics** agree
  exactly.
* mpsplat on **CPU** and mpsplat on **MPS** differ from the reference by the same numbers to 6 decimal
  places, so the Metal kernels carry no additional error over the pure-torch path.
