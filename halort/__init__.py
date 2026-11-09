"""
halort - minimal HaloRT / Denoiser / DenoiserX model library.

Quick start:
    import halort

    net = halort.create_scatnet('halort', cameras=1, width=512, height=512)
    # pretrained weights are loaded automatically from the files bundled
    # inside the installed package (see PRETRAINED_DIR below) - pass
    # load_pretrained=False to skip this and load your own checkpoint(s).

    net.set_extinction(cloud_tensor, density_scale=400.0)
    net.set_scattering_albedo((0.99, 0.99, 0.99))
    net.set_anisotropy(0.3)
    net.set_environment(environment_tensor)
    net.set_camera(0, origin, target, up)
    net.set_light(light_direction=(0.0, 1.0, 0.0), light_intensity=(1.0, 1.0, 1.0))

    radiance = net.reconstruct_radiance_fused()   # (1, H, W, 3)

See notebooks/ for complete Colab-ready examples, including scene setup
from scratch (cloud grid, HDR environment) and all three model types.

Needs the `rdv` package (Vulkan-CUDA interop rendering engine, a separate
project) importable - see the top-level README for setup.
"""
import os

from ._modeling import ScatNet, DenoiserScatNet, DenoiserXScatNet, HaloScatNet2, HaloRTCompose

__all__ = [
    "ScatNet", "DenoiserScatNet", "DenoiserXScatNet", "HaloScatNet2", "HaloRTCompose",
    "create_scatnet", "pretrained_path", "PRETRAINED_DIR",
]

# Bundled as package data (see pyproject.toml's package-data entry for
# "pretrained/*.pt") - pip install (including `pip install git+...`, no
# repo clone needed) copies this directory alongside the installed code,
# so it resolves correctly regardless of install method or cwd.
PRETRAINED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained")


def pretrained_path(filename: str) -> str:
    """Absolute path to a bundled pretrained weight file (see
    PRETRAINED_DIR) - e.g. pretrained_path('halo_dir.pt')."""
    return os.path.join(PRETRAINED_DIR, filename)


_MODEL_CLASSES = {
    "denoiser": DenoiserScatNet,
    "denoiserx": DenoiserXScatNet,
    "halort": HaloScatNet2,
}
# bundled checkpoint filename(s) per model - HaloRT needs the (dir, env) pair
_PRETRAINED_FILES = {
    "denoiser": ("denoiser.pt",),
    "denoiserx": ("denoiserx.pt",),
    "halort": ("halo_dir.pt", "halo_env.pt"),
}


def create_scatnet(model: str, cameras: int = 1, width: int = 512, height: int = 512,
                    jittered: bool = False, load_pretrained: bool = True, **kwargs):
    """Builds one of the three supported model types:

      - 'denoiser'  -> DenoiserScatNet   (small autoencoder, 9ch guide)
      - 'denoiserx' -> DenoiserXScatNet  (fuller autoencoder, 13ch guide)
      - 'halort'    -> HaloScatNet2      (two HaloNetFast UNets - dir/env)

    By default (load_pretrained=True) the model's bundled checkpoint(s)
    under PRETRAINED_DIR are loaded immediately - the returned net is
    ready to render. Pass load_pretrained=False to get an untrained net
    and call net.load_pretrained(...) yourself with a custom checkpoint
    (see each class's own method for its exact argument(s): HaloScatNet2
    takes (dir_path, env_path), Denoiser(X)ScatNet take a single path/
    state_dict).

    cameras/width/height/jittered are passed straight to the underlying
    ScatNet constructor (see halort/_modeling.py's own ScatNet.__init__).
    'halort' additionally accepts environment_samples (int, default 16) via
    kwargs, same as HaloScatNet2's own constructor.

    All three share the same scene-setup API (set_extinction,
    set_scattering_albedo, set_anisotropy, set_environment, set_camera,
    set_light) since they all inherit it unchanged from ScatNet - only the
    model-evaluation/output methods differ per class (see each class's own
    docstring: capture_condition / capture_reconstructed_radiance for
    Denoiser(X), capture_condition / reconstruct_radiance /
    reconstruct_radiance_fused for HaloRT)."""
    key = model.lower()
    if key not in _MODEL_CLASSES:
        raise ValueError(f"unknown model {model!r} - expected one of {sorted(_MODEL_CLASSES)}")
    cls = _MODEL_CLASSES[key]
    net = cls(cameras, width, height, jittered, **kwargs).cuda()
    if load_pretrained:
        paths = [pretrained_path(f) for f in _PRETRAINED_FILES[key]]
        net.load_pretrained(*paths)
    return net
