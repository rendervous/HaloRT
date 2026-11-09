/*
Like shaders/volumes/volume_profile_dda.h (rdv_project's own
VolumeProfileDDA - same majorant-grid DDA primary-ray marching, same
depth/tr/nee/sa/g/S feature layout), but:

  - the environment NEE term imports rdv's own EnvironmentSamplerTrait
    (importance-sampling the environment's own brightness distribution -
    e.g. XREnvironmentSampler - then evaluating the phase function at the
    sampled direction) instead of VolumeProfileDDA's own RIS-over-phase-
    samples (importance-sampling the phase function, then evaluating the
    environment directly) - see pbvr_env.h's own NEE block for the exact
    same pattern this mirrors.
  - the trailing light_intensity output block is dropped entirely (see
    VolumeProfileDDA's own shader comment: "not used as input of the
    model") - VolumeDeepProfile's own OUT is 4 + SPECTRAL_DIM*2 +
    DEPTH_SAMPLES (two SPECTRAL_DIM-sized blocks - env, sa - instead of
    VolumeProfileDDA's three), so _output[OUT - 1] = S (the interaction
    probability, set once, near the top, from OUT itself rather than a
    SPECTRAL_DIM-relative offset) lands in the right place with no other
    change needed.

Clip plane (clip_direction/clip_distance, world space): only the medium where
dot(clip_direction, x) >= clip_distance is considered - the primary ray's
[tMin, tMax] and every NEE shadow ray's segment are cut to that half-space, so
densities below the plane neither scatter nor attenuate. A zero clip_direction
disables it.
*/
#ifndef RDV_DEEP_PROFILE_CLIP
#define RDV_DEEP_PROFILE_CLIP
// Cuts [t0, t1] along x + w*t to the half-space dot(n, p) >= d (any space, as
// long as n/d/x/w agree). A ray parallel to the plane is kept or removed whole.
void rdv_clip_segment(vec3 n, float d, vec3 x, vec3 w, inout float t0, inout float t1)
{
    float s0 = dot(n, x) - d;
    float denom = dot(n, w);
    if (abs(denom) < 1e-12)
    {
        if (s0 < 0.0)
            t1 = t0 - 1.0; // entirely below the plane: empty segment
        return;
    }
    float tp = -s0 / denom;
    if (denom > 0.0)
        t0 = max(t0, tp); // entering the kept half-space at tp
    else
        t1 = min(t1, tp); // leaving it at tp
}
#endif

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
    // clip plane in object space: p_world = mat3(M) p + O, so dot(n, p_world) >= d
    // becomes dot(transpose(mat3(M)) n, p) >= d - dot(n, O); t is the same in both spaces
    vec3 clip_dir = deref(node).clip_direction;
    bool clip_on = dot(clip_dir, clip_dir) > 0.0;
    vec3 clip_n = transpose(mat3(M)) * clip_dir;
    float clip_d = deref(node).clip_distance - dot(clip_dir, O);
    // notice, in this point wo is not normalized, but is not a problem since we will use it for traversing the unnormalized density field, and we will normalize it later for sampling the phase function

    [[unroll]]
    for (int i=0; i < OUT; i++)
        _output[i] = 0.0; // transmittance 1 till infinity
    [[unroll]]
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[DEPTH_SAMPLES + 3 + SPECTRAL_DIM + i] = 1.0;
    _output[OUT - 1] = 0.0;
    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    tMin = max(0, tMin);
    if (clip_on)
        rdv_clip_segment(clip_n, clip_d, x, wo, tMin, tMax);
    if (tMax <= 0 || tMin >= tMax)
    return; // ray points away from the volume
    // Cell-by-cell DDA traversal of majorant_grid - see
    // VolumeProfileDDA's own shader for the full rationale, unchanged
    // here.
    ivec3 dim = ivec3(
        deref(deref(node).majorant_grid).shape[2],
        deref(deref(node).majorant_grid).shape[1],
        deref(deref(node).majorant_grid).shape[0]
    );
    vec3 cell_size = vec3(2.0) / vec3(dim);
    vec3 entry_pos = x + wo * tMin;
    ivec3 entry_cell = clamp(ivec3(floor((entry_pos * 0.5 + 0.5) * vec3(dim))), ivec3(0), dim - ivec3(1));
    wo.x += float(wo.x == 0.0)*0.0000001;
    wo.y += float(wo.y == 0.0)*0.0000001;
    wo.z += float(wo.z == 0.0)*0.0000001;
    ivec3 dda_step = ivec3(sign(wo));
    vec3 inv_wo = 1.0 / wo; // a 0 component makes that axis' own tMaxAxis/tDelta +-inf below (never the smallest), same robust idiom math_library.h's ray_box_intersection relies on
    vec3 next_boundary = (vec3(entry_cell) + max(vec3(dda_step), vec3(0.0))) * cell_size - vec3(1.0);
    vec3 entry_tMaxAxis = (next_boundary - x) * inv_wo;
    vec3 tDelta = cell_size * abs(inv_wo);

    // compute tau
    float tau = .0;
    // First t (along the ray, from tMin) where a real, non-zero density
    // sample was found during this pass, together with the exact DDA
    // cell/tMaxAxis state the walk below had at that same point - the
    // second traversal only ever needs to start from here instead of
    // re-walking the guaranteed-zero-contribution empty prefix from tMin
    // again (extinction==0 the whole way there, by definition of "first" -
    // skipping it changes nothing about current_tau/the depth samples
    // found, just the number of wasted fine-step iterations before the
    // first real one). The cell/tMaxAxis state is snapshotted directly
    // from this walk's own DDA variables rather than recomputed from
    // first_nonempty_t via floor() afterward: recomputing independently is
    // NOT guaranteed to agree with the incrementally-tracked DDA cell when
    // first_nonempty_t lands on or near a cell boundary (floating-point
    // rounding can put floor() one cell off from where the DDA walk
    // actually was), which silently sends pass 2 into the wrong majorant
    // cell for a small fraction of rays - confirmed by a direct old-vs-new
    // output comparison (verify_deep_profile_fix.py / diff_deep_profile_fix.py)
    // showing exact agreement on every deterministic channel EXCEPT a small
    // fraction of outlier pixels, consistent with this boundary case.
    float first_nonempty_t = -1.0;
    ivec3 first_nonempty_cell = ivec3(0);
    vec3 first_nonempty_tMaxAxis = vec3(0.0);
    {
        ivec3 cell = entry_cell;
        vec3 tMaxAxis = entry_tMaxAxis;
        float t = tMin;
        for (int guard = 0; guard < 4096 && t < tMax; guard++)
        {
            float cell_exit = min(min(tMaxAxis.x, tMaxAxis.y), min(tMaxAxis.z, tMax));
            float cell_majorant[1];
            forward(deref(node).majorant_grid, int[3](cell.z, cell.y, cell.x), cell_majorant);
            if (cell_majorant[0] > 0.0)
            {
                for (int safety = 0; t < cell_exit && safety < 4096; safety++)
                {
                    float extinction_val = extinction(node, x + wo * t) * deref(node).density_scale; // sample extinction at the middle of the step for better accuracy
                    if (first_nonempty_t < 0.0 && extinction_val > 0.0)
                    {
                        first_nonempty_t = t;
                        first_nonempty_cell = cell;
                        first_nonempty_tMaxAxis = tMaxAxis;
                    }
                    tau += extinction_val * deref(node).step_size;
                    t += deref(node).step_size;
                    if (tau > 20.0)
                        break; // transmittance is too low, can stop early
                }
                if (tau > 20.0)
                    break;
            }
            t = cell_exit;
            if (tMaxAxis.x <= tMaxAxis.y && tMaxAxis.x <= tMaxAxis.z)
                { cell.x += dda_step.x; tMaxAxis.x += tDelta.x; }
            else if (tMaxAxis.y <= tMaxAxis.z)
                { cell.y += dda_step.y; tMaxAxis.y += tDelta.y; }
            else
                { cell.z += dda_step.z; tMaxAxis.z += tDelta.z; }
                if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, dim)))
                    break; // walked off the majorant grid
        }
    }

    float current_tau = 0.0;
    int current_depth_index  = 0;
    float scale = deref(node).transmittance_threshold[current_depth_index];
    float tau_crossing = scale * tau < 0.01 ? 0.5 * scale * tau : scale * tau >= 10.0 ? 1 : 1 - (scale * tau * exp(-scale * tau))/(1 - exp(-scale * tau)); // expected tau when crossing the transmittance threshold, derived from the exponential distribution of free-flight distance
    if (tau > 0.0001)
    {
        // tau>0.0001 here guarantees first_nonempty_t/_cell/_tMaxAxis were
        // actually set above (pass 1 only accumulates tau from non-zero
        // extinction samples) - reuse that exact DDA state instead of
        // restarting the walk (and the majorant-grid cell traversal) from
        // tMin again (see first_nonempty_t's own comment above for why
        // this must be a carried-over snapshot, not a recomputation).
        ivec3 cell = first_nonempty_cell;
        vec3 tMaxAxis = first_nonempty_tMaxAxis;
        float t = first_nonempty_t; // restart raymarching from the first non-empty sample found in pass 1, not tMin
        bool done = false;
        for (int guard = 0; guard < 4096 && t < tMax && !done; guard++)
        {
            float cell_exit = min(min(tMaxAxis.x, tMaxAxis.y), min(tMaxAxis.z, tMax));
            float cell_majorant[1];
            forward(deref(node).majorant_grid, int[3](cell.z, cell.y, cell.x), cell_majorant);
            if (cell_majorant[0] > 0.0)
                for (int safety = 0; t < cell_exit && safety < 4096; safety++)
                {
                    float extinction_val = extinction(node, x + wo * t) * deref(node).density_scale; // sample extinction at the middle of the step for better accuracy
                    current_tau += extinction_val * deref(node).step_size;
                    for (int safety2 = 0; current_tau * scale >= tau_crossing && safety2 < DEPTH_SAMPLES + 1; safety2++)
                    {
                        _output[current_depth_index] = t;
                        current_depth_index++;
                        if (current_depth_index >= DEPTH_SAMPLES)
                            break;
                        scale = deref(node).transmittance_threshold[current_depth_index];
                        tau_crossing = scale * tau < 0.01 ? 0.5 * scale * tau : scale * tau >= 10 ? 1 : 1 - (scale * tau * exp(-scale * tau))/(1 - exp(-scale * tau));
                    }
                    if (current_depth_index >= DEPTH_SAMPLES)
                    {
                        done = true;
                        break;
                    }
                    t += deref(node).step_size;
                    if (current_tau > 20.0)
                    {
                        done = true;
                        break;
                    }
                }
            if (done)
                break;
            t = cell_exit;
            if (tMaxAxis.x <= tMaxAxis.y && tMaxAxis.x <= tMaxAxis.z)
                { cell.x += dda_step.x; tMaxAxis.x += tDelta.x; }
            else if (tMaxAxis.y <= tMaxAxis.z)
                { cell.y += dda_step.y; tMaxAxis.y += tDelta.y; }
            else
                { cell.z += dda_step.z; tMaxAxis.z += tDelta.z; }
                if (any(lessThan(cell, ivec3(0))) || any(greaterThanEqual(cell, dim)))
                    break; // walked off the majorant grid
        }
    }

    _output[OUT - 1] = 1 - exp(-tau); // Interaction probability - OUT-relative, so it lands at the end
    float S = _output[OUT - 1];

    // compute nee and light dir for the first sample
    if (current_depth_index > 0) // if we have at least one valid sample, we can compute nee and scattering for the first sample
    {
        float check_t = _output[deref(node).sample_index]; // index to the depth sample that we want to check for nee, scattering, etc.

        // reencode depths
        if (deref(node).use_depth_encoding == 1)
        {
            [[unroll]]
//                for (int i=0; i<DEPTH_SAMPLES; i++)
//                    _output[i] = 1.0 / (0.1 + _output[i]);
            for (int i=DEPTH_SAMPLES-1; i >= 1; i--)
                _output[i] = pow(max(0, _output[i] - _output[i-1]), 0.25);
            _output[0] = log (1 + 0.2*check_t);// 1 / (0.1 + max(0.0, check_t));
        }

        vec3 sample_pos = x + wo * check_t; // position of the last sample in object space
        float g = deref(node).g;
        if (deref(node).use_reduced_transmittance == 1)
            // feature: interaction on a reduced medium
            _output[DEPTH_SAMPLES] = 1 - exp(-0.2 * tau); //1 - (1 / (1 + tau));// 1 - exp(-0.2 * tau);
        else
            // feature: interaction on the original medium
            _output[DEPTH_SAMPLES] = S;

        if (deref(node).use_similarity_anisotropy == 1)
            // feature: general transmittance wrt similarity theory
            _output[DEPTH_SAMPLES + 1] = 1 - sign(g)*pow(abs(g), tau); //1 - exp(-(1-g) * tau);
        else
            _output[DEPTH_SAMPLES + 1] = g; // directly exposed

        float sa[SPECTRAL_DIM];
        scattering_albedo(node, sample_pos, sa);
        if (deref(node).use_scattering_accumulation == 1)
        {
            [[unroll]]
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 3 + SPECTRAL_DIM + i] = pow(sa[i], 2 * tau * (1 - sa[i]*g)); // estimated path-through absorption
        }
        else {
            [[unroll]]
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 3 + SPECTRAL_DIM + i] = sa[i];
        }

        vec3 light_dir = deref(node).light_direction;
        vec3 light_dir_object_space = L * light_dir; // convert light direction to object space
        ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
        if (clip_on) { float t0 = 0.0; rdv_clip_segment(clip_n, clip_d, sample_pos, light_dir_object_space, t0, tMax); }
        float RM_DISTANCE = 0.2;
        float nee_T;
        nee_T = tMax < 0 || tMax < tMin ? 1.0 : transmittance_rm(node, sample_pos, light_dir_object_space, min(tMax, RM_DISTANCE), deref(node).step_size, deref(node).density_scale);
        if (tMax > RM_DISTANCE && tMax >= tMin)
            nee_T *= transmittance_rt(node, sample_pos + light_dir_object_space*RM_DISTANCE, light_dir_object_space, tMax - RM_DISTANCE, deref(node).density_scale);
//            nee_T = transmittance_rt(node, sample_pos, light_dir_object_space, tMax, deref(node).density_scale);

        _output[DEPTH_SAMPLES + 2] = nee_T;// 4 * pi * nee_T * hg_phase_eval(w, light_dir, g); // direct light: probability * transmittance toward the fixed light (no phase/albedo weighting, same as VolumeProfileDDA's own feature)

        // Environment NEE, proper importance sampling this time:
        // sample a direction from the environment's own brightness
        // distribution (e.g. XREnvironmentSampler - see
        // EnvironmentSamplerTrait), evaluate the phase function at
        // that direction (env[] is already env(dir)/pdf(dir)
        // weighted - see environment_sampler_trait.h - so this is
        // exactly pbvr_env.h's own NEE block, just averaged over
        // environment_samples independent draws instead of one).
        int env_samples = deref(node).environment_samples;
        for (int s=0; s < env_samples; s++)
        {
            float env[SPECTRAL_DIM];
            light_dir = environment_sampler(node, sample_pos, w, env);
            //float phase = hg_phase_eval(w, light_dir, g);
            light_dir_object_space = L * light_dir; // convert light direction to object space
            ray_box_intersection(sample_pos, light_dir_object_space, tMin, tMax);
            if (clip_on) { float t0 = 0.0; rdv_clip_segment(clip_n, clip_d, sample_pos, light_dir_object_space, t0, tMax); }
            nee_T = (tMax <= 0.0 ? 1.0 : transmittance_rt(node, sample_pos, light_dir_object_space, tMax, deref(node).density_scale)) * hg_phase_eval(w, light_dir, g);
            [[unroll]]
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 3 + i] += nee_T * env[i];
        }
        if (env_samples > 0)
            [[unroll]]
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 3 + i] /= (env_samples);// * 4 * pi);
    }
}
