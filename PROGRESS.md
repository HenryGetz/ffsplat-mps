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
