"""
halort._profiles - minimal profiler/feature-map definitions needed by
Denoiser (FSGProfile), DenoiserX (VolumeDeepProfileNoNEE), and HaloRT
(VolumeDeepProfile), extracted from direct_neural_rendering's own
rdv_nr/_profiles.py. PBVRTPort (the shared noisy-radiance condition input
for Denoiser/DenoiserX) is also included. Older/ablation-only profiler
variants (ProfileFDTNEX7Port, VolumeDeepProfile2) were dropped.
"""

from rdv import *



class VolumeDeepProfile(Map,
                    RatiotrackingTransmittanceTrait, RaymarchingTransmittanceTrait,
                    ScatteringAlbedoTrait, EnvironmentSamplerTrait,
                    Generic[Generics.DEPTH_SAMPLES, Generics.MG]):
    __code__ = 'volume_deep_profile.h'
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    density_scale: float
    transform: mat4x3
    light_direction: vec3
    g: float
    transmittance_threshold: Array[Generics.DEPTH_SAMPLES, float]
    sample_index: int
    environment_samples: int
    step_size: float
    majorant_grid: Generics.MG
    use_depth_encoding: int
    use_reduced_transmittance: int
    use_similarity_anisotropy: int
    use_scattering_accumulation: int
    clip_direction: vec3
    """World-space clip plane normal: only medium where dot(clip_direction, x) >=
    clip_distance (above the plane) is considered - for the primary ray and the
    NEE shadow rays alike. A zero vector (the default) disables clipping."""
    clip_distance: float

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 majorant_grid: Generics.MG,
                 environment_sampler: Generics.ENVIRONMENT_SAMPLER,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 transmittance_threshold: list[float],
                 density_scale: float = 1.0,
                 transform: mat4x3 | None = None,
                 light_direction=(0.0, 1.0, 0.0),
                 g: float = 0.0,
                 step_size: float = 0.001,
                 environment_samples: int = 4,
                 sample_index: int = 0,
                 use_depth_encoding: bool = False,
                 use_reduced_transmittance: bool = False,
                 use_similarity_anisotropy: bool = False,
                 use_scattering_accumulation: bool = False,
                 clip_direction=(0.0, 0.0, 0.0),
                 clip_distance: float = 0.0,
                 ) -> None:
        super().__init__(extinction=extinction, majorant=majorant, environment_sampler=environment_sampler,
                         scattering_albedo=scattering_albedo)
        depth_samples = len(transmittance_threshold)
        self.match(Generics.IN, 6)
        self.match(Generics.DEPTH_SAMPLES, depth_samples)
        spectral_dim: int = Generics.SPECTRAL_DIM(self)
        # OUT
        # depth (1), deltas (2), scaled_transmittance (1), effective_transmittance (1),
        # nee (1), effective_albedo(S), reduced_transmitted_environment (S), real_transmittance (not used in the model)
        self.match(Generics.OUT, None if spectral_dim is None else 4 + spectral_dim*2 + depth_samples)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.light_direction = vec3(*light_direction) if not isinstance(light_direction, vec3) else light_direction
        self.density_scale = density_scale
        self.g = g
        self.step_size = step_size
        self.environment_samples = environment_samples
        self.sample_index = sample_index
        self.transmittance_threshold = transmittance_threshold
        majorant_grid = majorant_grid.cast(Grid[3, 1])
        self.match(Generics.MG, type(majorant_grid))
        self.majorant_grid = majorant_grid
        self.use_depth_encoding = int(use_depth_encoding)
        self.use_reduced_transmittance = int(use_reduced_transmittance)
        self.use_similarity_anisotropy = int(use_similarity_anisotropy)
        self.use_scattering_accumulation = int(use_scattering_accumulation)
        self.clip_direction = vec3(*clip_direction) if not isinstance(clip_direction, vec3) else clip_direction
        self.clip_distance = clip_distance


class VolumeDeepProfileNoNEE(Map,
                    RatiotrackingTransmittanceTrait, RaymarchingTransmittanceTrait,
                    ScatteringAlbedoTrait,
                    Generic[Generics.DEPTH_SAMPLES, Generics.MG]):
    """Same depth/tr/g/sa feature computation as VolumeDeepProfile (same
    two-pass majorant-grid tau walk, same use_depth_encoding/use_reduced_
    transmittance/use_similarity_anisotropy/use_scattering_accumulation
    flags), with the direct-light NEE and environment NEE blocks removed
    entirely - see volume_deep_profile_no_nee.h's own header comment. Used
    as DenoiserXScatNet's auxiliary guide feature buffer (rdv_nr/_modeling.py)
    in place of the old ProfileFDTNEX7Port - no majorant/environment_sampler/
    light_direction dependency needed at all, unlike VolumeDeepProfile.

    OUT = DEPTH_SAMPLES + SPECTRAL_DIM + 3."""
    __code__ = 'volume_deep_profile_no_nee.h'
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    density_scale: float
    transform: mat4x3
    g: float
    transmittance_threshold: Array[Generics.DEPTH_SAMPLES, float]
    sample_index: int
    step_size: float
    majorant_grid: Generics.MG
    use_depth_encoding: int
    use_reduced_transmittance: int
    use_similarity_anisotropy: int
    use_scattering_accumulation: int

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 majorant_grid: Generics.MG,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 transmittance_threshold: list[float],
                 density_scale: float = 1.0,
                 transform: mat4x3 | None = None,
                 g: float = 0.0,
                 step_size: float = 0.001,
                 sample_index: int = 0,
                 use_depth_encoding: bool = False,
                 use_reduced_transmittance: bool = False,
                 use_similarity_anisotropy: bool = False,
                 use_scattering_accumulation: bool = False,
                 ) -> None:
        super().__init__(extinction=extinction, majorant=majorant, scattering_albedo=scattering_albedo)
        depth_samples = len(transmittance_threshold)
        self.match(Generics.IN, 6)
        self.match(Generics.DEPTH_SAMPLES, depth_samples)
        spectral_dim: int = Generics.SPECTRAL_DIM(self)
        self.match(Generics.OUT, None if spectral_dim is None else depth_samples + spectral_dim + 3)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.density_scale = density_scale
        self.g = g
        self.step_size = step_size
        self.sample_index = sample_index
        self.transmittance_threshold = transmittance_threshold
        majorant_grid = majorant_grid.cast(Grid[3, 1])
        self.match(Generics.MG, type(majorant_grid))
        self.majorant_grid = majorant_grid
        self.use_depth_encoding = int(use_depth_encoding)
        self.use_reduced_transmittance = int(use_reduced_transmittance)
        self.use_similarity_anisotropy = int(use_similarity_anisotropy)
        self.use_scattering_accumulation = int(use_scattering_accumulation)


class PBVRTPort(Map,
                RatiotrackingTransmittanceTrait, ScatteringAlbedoTrait,
                EnvironmentTrait):
    """
    Exact-formulation port of rendervous.project's own historical pbvrt.h -
    see pbvrt_port.h. Per-bounce NEE is against a single fixed directional
    light (light_direction/light_intensity), matching the current
    framework's own PBVRT - NOT the old file's own literal
    `environment_sampler` call, which in every actual rendervous.project
    training/rendering script was always fed a single precomputed
    direction+value via a ConstantMap (never a live per-query sampler), i.e.
    was always a delta light in practice. Kept byte-faithful (to that actual
    behavior, not the literal old call signature) for reuse with pretrained
    checkpoints trained against this exact old radiance estimator (the
    Denoiser/DenoiserX "noisy radiance" input term) - see pbvrt_port.h's own
    docstring for the full rationale.

    OUT = SPECTRAL_DIM + 1: [0..SPECTRAL_DIM) radiance, [SPECTRAL_DIM] a
    "still background" flag (1.0 until the first scattering event, 0.0
    after).
    """
    __code__ = 'pbvrt_port.h'
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    transform: mat4x3
    light_direction: vec3
    light_intensity: vec3
    g: float
    density_scale: float

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 environment: Generics.ENVIRONMENT,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 transform: mat4x3 | None = None,
                 light_direction=(0.0, 1.0, 0.0),
                 light_intensity=None,
                 density_scale: float = 1.0,
                 g: float = 0.0,
                 ) -> None:
        super().__init__(extinction=extinction, majorant=majorant, environment=environment,
                          scattering_albedo=scattering_albedo)
        self.match(Generics.IN, 6)
        spectral_dim: int = Generics.SPECTRAL_DIM(self)
        self.match(Generics.OUT, None if spectral_dim is None else spectral_dim + 1)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.light_direction = vec3(*light_direction) if not isinstance(light_direction, vec3) else light_direction
        # vec3 (not Array[SPECTRAL_DIM, float]) - light_direction (same
        # 3-component shape, right above) demonstrably updates correctly
        # every frame via this same per-instance node assignment; the old
        # Array-typed declaration here was never actually reaching the
        # compiled shader after construction, silently leaving every
        # training instance's "noisy radiance" condition rendered at the
        # class default (1,1,1) regardless of the per-instance light_scale
        # passed to set_light() - see _denoiser_channel_training.py's own
        # note on this fix.
        self.light_intensity = vec3(1.0, 1.0, 1.0) if light_intensity is None else vec3(*light_intensity)
        self.density_scale = density_scale
        self.g = g


class FSGProfile(Map, RatiotrackingTransmittanceTrait, RaymarchingTransmittanceTrait,
                  ScatteringAlbedoTrait, Generic[Generics.MG]):
    """
    Exact-formulation port of old_profiles/profile_fsg.h ("first scattering
    guess") - see rdv_nr/fsg_profile.h. OUT = 1 (depth) + SPECTRAL_DIM
    (scattering albedo) + 1 (g) + 1 (S - binary 0/1 hit indicator, not in
    the original file, added back for DenoiserScatNet's own blending - see
    fsg_profile.h's own note).
    """
    __code__ = 'fsg_profile.h'
    __requires__ = [
        RandomLibrary,
        MathLibrary,
    ]
    density_scale: float
    transform: mat4x3
    g: float
    majorant_grid: Generics.MG

    def __init__(self,
                 extinction: Generics.EXTINCTION,
                 majorant: Generics.MAJORANT,
                 majorant_grid: Generics.MG,
                 scattering_albedo: Generics.SCATTERING_ALBEDO,
                 density_scale: float = 1.0,
                 transform: mat4x3 | None = None,
                 g: float = 0.0,
                 ) -> None:
        super().__init__(extinction=extinction, majorant=majorant, scattering_albedo=scattering_albedo)
        self.match(Generics.IN, 6)
        spectral_dim: int = Generics.SPECTRAL_DIM(self)
        self.match(Generics.OUT, None if spectral_dim is None else spectral_dim + 3)
        self.transform = mat4x3(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) \
            if transform is None else mat4x3(transform)
        self.density_scale = density_scale
        self.g = g
        majorant_grid = majorant_grid.cast(Grid[3, 1])
        self.match(Generics.MG, type(majorant_grid))
        self.majorant_grid = majorant_grid

