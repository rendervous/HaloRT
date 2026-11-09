# HaloRT - Halo-attention-based Radiative Transfer model
Source code of the paper Real-Time Volumetric Light Transport Inference without a Path

This repository is a **minimal, standalone package** for the three neural
volumetric-scattering models from the HaloRT project:

- **HaloRT** - the paper's own real-time model (two attention-based UNets,
  one for direct/NEE-conditioned scattering, one for environment
  scattering).
- **Denoiser** / **DenoiserX** - lightweight single-autoencoder baselines
  that denoise a few-sample noisy path-traced render, guided by auxiliary
  scene features (DenoiserX uses a fuller feature set and is the stronger
  of the two).

It only contains what's needed to *build, load, and run* these three
models against a scene (cloud density grid + HDR environment + light) -
scene setup, condition capture, and radiance reconstruction. It does not
include training code, the full experiment suite, or the dataset
generation pipeline from the original research repository.

## Install

This package itself has no native code, but it depends on
[`rendervous`](https://github.com/rendervous/rdv_project) (`rdv`), the
Vulkan-CUDA interop rendering engine these models are built on, which in
turn depends on [`vulky`](https://github.com/rendervous/vk_project) (`vk`)
for raw Vulkan device/resource access. Both are published on PyPI with
prebuilt wheels, so on a machine with a Vulkan-capable GPU:

```bash
pip install rendervous   # pulls in vulky automatically
pip install git+https://github.com/rendervous/HaloRT.git
```

or, for local development:

```bash
git clone https://github.com/rendervous/HaloRT.git
cd HaloRT
pip install -e .
```

See `notebooks/` for a complete Colab-ready walkthrough, including
installing on a fresh Colab runtime.

## Quick start

```python
import torch
import rdv
import halort

rdv.device(0)

net = halort.create_scatnet("halort", width=512, height=512)
net.load_pretrained("pretrained/halo_dir.pt", "pretrained/halo_env.pt")

# scene setup - same API for all three model types (inherited from ScatNet)
net.set_extinction(cloud_density_grid, density_scale=400.0)   # (D,H,W) or (D,H,W,1) torch tensor
net.set_scattering_albedo((0.99, 0.99, 0.99))
net.set_anisotropy(0.3)                                        # Henyey-Greenstein g
net.set_environment(hdr_environment_tensor)                    # (H,W,3) torch tensor
net.set_camera(0, origin=(0, 0.5, 2.5), target=(0, 0, 0), up=(0, 1, 0))
net.set_light(light_direction=(0, 1, 0), light_intensity=(1, 1, 1))

with torch.no_grad():
    radiance = net.reconstruct_radiance_fused()   # (1, H, W, 3)
```

`create_scatnet(model=...)` accepts `'denoiser'`, `'denoiserx'`, or
`'halort'`. Denoiser/DenoiserX use `net.load_pretrained("pretrained/denoiser.pt")`
(single file) and `net.capture_reconstructed_radiance()` instead.

## Pretrained weights

`pretrained/` holds bare `state_dict`s only (no optimizer/scheduler/step
state from training) - small (~4.5 MB each):

| File | Model |
| --- | --- |
| `halo_dir.pt` + `halo_env.pt` | HaloRT (both required together) |
| `denoiser.pt` | Denoiser |
| `denoiserx.pt` | DenoiserX |

## Cloud volume datasets

A reference to a Hugging Face dataset with additional cloud volumes
suitable for rendering with these models will be added here.

## License

MIT - see `LICENSE`.
