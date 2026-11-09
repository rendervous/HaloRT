/*
Identical DDA depth/tr/g/sa feature computation to volume_deep_profile.h
(same two-pass majorant-grid tau walk, same use_depth_encoding/
use_reduced_transmittance/use_similarity_anisotropy/use_scattering_
accumulation flags), but with the direct-light NEE and environment NEE
blocks removed entirely - this variant is used as DenoiserX's auxiliary
guide feature buffer (see rdv_nr/_modeling.py's DenoiserXScatNet), which
only ever fed those extra channels forward into the Autoencoder, never
consumed them for anything environment-sampler-specific; dropping them
saves the environment-importance-sampling dispatch cost entirely and
needs no majorant/environment_sampler/light_direction dependency at all.

OUT = DEPTH_SAMPLES + SPECTRAL_DIM + 3 (4 + SPECTRAL_DIM*2 + DEPTH_SAMPLES
from volume_deep_profile.h, minus the 1 (nee) + SPECTRAL_DIM (env) slots
removed here). Output layout: [0..DEPTH_SAMPLES-1] depth samples |
[DEPTH_SAMPLES] tr | [DEPTH_SAMPLES+1] g | [DEPTH_SAMPLES+2 ..
DEPTH_SAMPLES+1+SPECTRAL_DIM] sa | [OUT-1] S (interaction probability).
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
    // notice, in this point wo is not normalized, but is not a problem since we will use it for traversing the unnormalized density field, and we will normalize it later for sampling the phase function

    [[unroll]]
    for (int i=0; i < OUT; i++)
        _output[i] = 0.0; // transmittance 1 till infinity
    [[unroll]]
    for (int i=0; i<SPECTRAL_DIM; i++)
        _output[DEPTH_SAMPLES + 2 + i] = 1.0;
    _output[OUT - 1] = 0.0;
    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    tMin = max(0, tMin);
    if (tMax <= 0 || tMin >= tMax)
    return; // ray points away from the volume
    // Cell-by-cell DDA traversal of majorant_grid - see
    // volume_deep_profile.h's own shader for the full rationale, unchanged
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
    // cell/tMaxAxis state the walk below had at that same point - see
    // volume_deep_profile.h's own comment on first_nonempty_t for the full
    // rationale (unchanged here).
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
        // tMin again.
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

    // compute depth re-encoding, tr, g and sa for the first sample - no
    // NEE/environment term at all here (see this file's own header comment).
    if (current_depth_index > 0) // if we have at least one valid sample, we can compute scattering features for the first sample
    {
        float check_t = _output[deref(node).sample_index]; // index to the depth sample that we want to check for scattering, etc.

        // reencode depths
        if (deref(node).use_depth_encoding == 1)
        {
            [[unroll]]
            for (int i=DEPTH_SAMPLES-1; i >= 1; i--)
                _output[i] = pow(max(0, _output[i] - _output[i-1]), 0.25);
            _output[0] = log (1 + 0.2*check_t);
        }

        vec3 sample_pos = x + wo * check_t; // position of the last sample in object space
        float g = deref(node).g;
        if (deref(node).use_reduced_transmittance == 1)
            // feature: interaction on a reduced medium
            _output[DEPTH_SAMPLES] = 1 - exp(-0.2 * tau);
        else
            // feature: interaction on the original medium
            _output[DEPTH_SAMPLES] = S;

        if (deref(node).use_similarity_anisotropy == 1)
            // feature: general transmittance wrt similarity theory
            _output[DEPTH_SAMPLES + 1] = 1 - sign(g)*pow(abs(g), tau);
        else
            _output[DEPTH_SAMPLES + 1] = g; // directly exposed

        float sa[SPECTRAL_DIM];
        scattering_albedo(node, sample_pos, sa);
        if (deref(node).use_scattering_accumulation == 1)
        {
            [[unroll]]
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 2 + i] = pow(sa[i], 2 * tau * (1 - sa[i]*g)); // estimated path-through absorption
        }
        else {
            [[unroll]]
            for (int i=0; i<SPECTRAL_DIM; i++)
                _output[DEPTH_SAMPLES + 2 + i] = sa[i];
        }
    }
}
