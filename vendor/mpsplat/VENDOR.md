# Vendored: mpsplat

This directory is a **vendored copy** of [zacsmms/mpsplat](https://github.com/zacsmms/mpsplat),
the Apple-Silicon (MPS / Metal) port of [gsplat](https://github.com/nerfstudio-project/gsplat).
It replaces the `gsplat==1.4.0` CUDA-only provider in `pyproject.toml`.

| | |
|---|---|
| Upstream | `https://github.com/zacsmms/mpsplat` |
| Pinned commit | `94c89496f99d74d81e2a298e1f6159cc3978af34` |
| Commit date | 2026-05-12 ("Add RETURN_TO.md — paused state of the TSDF-prior experiment") |
| Vendored version | `gsplat` **1.5.3** (distribution name `mpsplat`) |
| License | Apache-2.0 (`LICENSE`, unchanged from upstream) |

No file in here was modified: it is a byte-for-byte copy of the tracked files listed below at the
commit above. Verify with:

```bash
git clone https://github.com/zacsmms/mpsplat /tmp/mpsplat-check
git -C /tmp/mpsplat-check checkout 94c89496f99d74d81e2a298e1f6159cc3978af34
diff -r --exclude=VENDOR.md /tmp/mpsplat-check/gsplat vendor/mpsplat/gsplat   # no differences
```

## Why the package is still imported as `gsplat`

mpsplat deliberately keeps the upstream API and the upstream *package* name (`gsplat`); it only
renames the distribution to `mpsplat`. `pyproject.toml` therefore depends on the distribution
`mpsplat` and points it at this directory with a uv *editable* path source, so `import gsplat`
resolves to the copy in this directory — no runtime dependence on the live upstream repository.

Editable (rather than a built wheel) is deliberate: upstream's `MANIFEST.in` does not list
`gsplat/mps/_shaders/*.metal`, so a non-editable build could omit the 14 Metal shader sources.
An editable install reads them straight from this tree.

## Contents

Copied from the pinned commit, unmodified:

```
gsplat/            the library (includes gsplat/mps/_shaders/*.metal — 14 files)
LICENSE            Apache-2.0
README.md          upstream readme (attribution + status of the port)
CITATION.bib       upstream citation
setup.py           build metadata (name=mpsplat, version=1.5.3)
pyproject.toml     build-system table
MANIFEST.in
```

Deliberately **not** vendored (not needed to `import gsplat`, and large/binary): `docs/` (25 MB,
includes MP4s), `assets/`, `media/`, `notebooks/`, `profiling/`, `examples/`, `tests/`, `methods/`,
`QUICKSTART.md`, `EXPLORATION.md`, `formatter.sh`, `pytest.ini`, `.github/`, `mpsplat.egg-info/`.
