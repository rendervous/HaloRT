/*
Exact-formulation port of rendervous.project/rdv/include/volume_rendering/
pbvrt.h - the ACTUAL historical PBVRT the Denoiser/DenoiserX pretrained
checkpoints were trained against.

Per-bounce NEE is against a single fixed directional light (light_direction/
light_intensity fields), matching the current framework's own PBVRT - NOT
the old file's own literal `forward(parameters.environment_sampler, ...)`
call, which looked like environment importance-sampling but, in every
actual training/rendering script in rendervous.project, was always fed a
rdv.ConstantMap of one precomputed direction+value per scene (never a live
per-query sampler) - i.e. was ALWAYS a delta light in practice, just spelled
as an "environment_sampler" call. A genuine live importance sampler here
(this codebase's own XREnvironmentSampler) changes the lighting model
itself - broad, resampled-every-bounce environment contribution instead of
a single delta contribution - which materially changes the noisy radiance
distribution these checkpoints were trained to denoise. Only the final
escape term (after the ray leaves the volume with no further collision)
samples the real environment texture directly, same as the original.

No majorant-grid DDA (old pbvrt.h didn't have one), no bounce cap in the
original (the guard/safety counter below is a defensive addition only, does
not change behavior in any case that would have terminated originally), g
read as a scalar field instead of a per-point submap (see the other
*_port.h files' own identical note: ScatNet has never carried a spatially-
varying anisotropy field, so this is the exact same value the old submap
call would have returned).

OUT = SPECTRAL_DIM + 1: [0..SPECTRAL_DIM) radiance, [SPECTRAL_DIM] a "still
background" flag - 1.0 until the first real scattering event, 0.0 from then
on (unchanged from the original's own semantics).

Loop/threshold mechanics (entry epsilon, majorant/maj_distance epsilons,
3-way min against the remaining segment distance, continue/break
conditions, safety-cap style) realigned to match pbvr_nee.h's own
(shaders/volumes/pbvr_nee.h) - a from-scratch, differently-coded but
formula-equivalent NEE estimator in the current framework, to rule out
whether this file's own particular control-flow shape (as opposed to the
underlying algorithm/output semantics, both left untouched: still no bounce
cap, still the SPECTRAL_DIM+1 background flag + light_intensity-weighted
NEE + post-loop environment escape term, none of which pbvr_nee.h has) was
responsible for a GPU-generation-specific timing regression.
*/
void forward(RDV_NODE node, float _input[6], out float _output[OUT])
{
    vec3 x = vec3(_input[0], _input[1], _input[2]);
    vec3 w = vec3(_input[3], _input[4], _input[5]);

    // Transform ray to local space
    mat4x3 M = deref(node).transform; // from object to world space
    mat3 L = inverse(mat3(M));
    vec3 O = M[3];
    x = L * (x - O); // convert world position to object space position (use same x)
    vec3 wo = L * w; // convert world direction to object space direction

    for (int i = 0; i < SPECTRAL_DIM; i++)
        _output[i] = 0.0;
    _output[SPECTRAL_DIM] = 1.0;

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    if (tMax <= 0 || tMin >= tMax) // ray points away from the volume
        return;
    tMin = max(0, tMin) + 1e-5;

    x += wo * tMin; // initial position in object space
    float d = tMax - tMin; // max_t wrt wo

    float g = deref(node).g;
    float density_scale = deref(node).density_scale;
    vec3 light_dir = normalize(deref(node).light_direction);

    float W[SPECTRAL_DIM]; // path throughput, accumulation is direct to output to save register, so we need to store it in a local variable
    for (int i = 0; i < SPECTRAL_DIM; i++)
        W[i] = 1.0;

    int bounces = 0;
    // Set right before the only early exit out of this loop (low
    // throughput) and checked below to gate the post-loop block - lets that
    // exit use `break` (falls through to this function's one shared exit
    // path) instead of an early `return` (a second, separate exit path),
    // while still producing byte-identical output to the old `return`:
    // the old code skipped the post-loop background-flag/environment-escape
    // block entirely on this path, so this flag reproduces exactly that -
    // see this file's own git history / the investigation that found the
    // early `return` itself (not the loop body, not the majorant/threshold
    // mechanics, not the light_intensity application site - all individually
    // ruled out first) was responsible for a Blackwell-driver-specific
    // register-allocation regression that this exact `return` vs `break`
    // distinction reproduces and removes on demand.
    bool terminated_early = false;
    for (int safety = 0; safety < 16384; safety++) { // safety: defensive only, see file header
        float maj_distance;
        float maj = majorant(node, x, wo, maj_distance) + 1e-5;
        if (maj_distance <= 0.00001)
            maj_distance = 0.00001;
        float dt = min(min(d, maj_distance), -log(1.0 - random()) / (density_scale * maj)); // sample free-flight distance
        x += dt * wo;
        d -= dt;
        if (dt >= maj_distance)
            continue; // no interaction withing slab, continue
        if (d <= 1e-7)
            break; // reached the end of the segment
        if (random() >= extinction(node, x) / maj) // null collision
            continue;

        // -- Valid collision --
        // modulate path-throughput with scattering albedo
        float sa[SPECTRAL_DIM];
        scattering_albedo(node, x, sa); // compute scattering albedo
        for (int i = 0; i < SPECTRAL_DIM; i++)
            W[i] *= sa[i];
        bounces++;

        bool some_throughput = false;
        for (int i = 0; i < SPECTRAL_DIM; i++)
            if (W[i] > 0.0001)
                some_throughput = true;
        if (!some_throughput) {
            terminated_early = true;
            break; // path throughput too low - break (not return), see terminated_early's own note above for why this is still byte-identical to the old behavior
        }

        // add contribution from the fixed directional light with NEE - a
        // delta light needs no importance sampling/pdf, its contribution is
        // just throughput * light_intensity * phase * shadow transmittance.
        float nee_phase = hg_phase_eval(w, light_dir, g);
        vec3 light_dir_object_space = L * light_dir; // convert to object space
        float nee_d;
        ray_box_intersection(x, light_dir_object_space, tMin, nee_d); // compute intersection with volume boundary
        float tr = transmittance_rt(node, x, light_dir_object_space, nee_d, density_scale);
        tr *= nee_phase;
        for (int i = 0; i < SPECTRAL_DIM; i++)
            _output[i] += W[i] * deref(node).light_intensity[i] * tr; // accumulate NEE contribution

        // scatter ray
        w = hg_phase_sample(w, g); // compute scattering direction in world space
        wo = L * w; // convert scattering direction to object space for next iteration
        // compute next distance
        ray_box_intersection(x, wo, tMin, d); // compute intersection with volume boundary for next iteration
    }

    if (bounces > 0 && !terminated_early) {
        _output[SPECTRAL_DIM] = 0.0; // assumed scattering
        float env[SPECTRAL_DIM];
        environment(node, w, env); // add environment contribution if ray escaped the volume
        for (int i = 0; i < SPECTRAL_DIM; i++)
            _output[i] += W[i] * env[i];
    }
}
