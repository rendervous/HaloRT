/*
Exact-formulation port of old_profiles/profile_fsg.h ("first scattering
guess" - depth, scattering albedo and anisotropy at the first real collision
along the ray), confirmed against the real historical source
(rendervous.project/rdv/include/volume_rendering/profile_fsg.h, byte-
identical to the copy in old_profiles/) and the actual training scripts
(train_denoiser.py): OUT = SPECTRAL_DIM + 3 (depth, albedo, g, S). The
trailing S (binary 0/1 - whether this DDA delta-tracking pass found an
accepted collision) is not in the original file - added back for
DenoiserScatNet's own combined-capture blending (radiance*S +
background*(1-S)), same purpose as VolumeDeepProfile's own S, but here a
single noisy 0/1 sample rather than a deterministic value (this profile
never computes a full-ray tau the way VolumeDeepProfile/ProfileFDTNEX7Port
do, so a deterministic 1-exp(-tau) isn't available cheaply here - a hit/miss
sample is what the original delta-tracking loop already produces for free).
Same delta-tracking algorithm as the original otherwise, only walking the
majorant_grid cell-by-cell (the same DDA idiom volume_deep_profile.h uses)
instead of the old single global-majorant step - same estimator either way
(unbiased delta-tracking), not a formulation change.
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

    [[unroll]]
    for (int i = 0; i < OUT; i++)
        _output[i] = 0.0;

    float tMin, tMax;
    ray_box_intersection(x, wo, tMin, tMax);
    tMin = max(0, tMin);
    if (tMax <= 0 || tMin >= tMax)
        return; // ray points away from the volume

    // Cell-by-cell DDA traversal of majorant_grid - see volume_deep_profile.h
    // for the full rationale, unchanged here.
    ivec3 dim = ivec3(
        deref(deref(node).majorant_grid).shape[2],
        deref(deref(node).majorant_grid).shape[1],
        deref(deref(node).majorant_grid).shape[0]
    );
    vec3 cell_size = vec3(2.0) / vec3(dim);
    vec3 entry_pos = x + wo * tMin;
    ivec3 cell = clamp(ivec3(floor((entry_pos * 0.5 + 0.5) * vec3(dim))), ivec3(0), dim - ivec3(1));
    wo.x += float(wo.x == 0.0) * 0.0000001;
    wo.y += float(wo.y == 0.0) * 0.0000001;
    wo.z += float(wo.z == 0.0) * 0.0000001;
    ivec3 dda_step = ivec3(sign(wo));
    vec3 inv_wo = 1.0 / wo;
    vec3 next_boundary = (vec3(cell) + max(vec3(dda_step), vec3(0.0))) * cell_size - vec3(1.0);
    vec3 tMaxAxis = (next_boundary - x) * inv_wo;
    vec3 tDelta = cell_size * abs(inv_wo);

    float t = tMin;
    for (int guard = 0; guard < 4096 && t < tMax; guard++)
    {
        float cell_exit = min(min(tMaxAxis.x, tMaxAxis.y), min(tMaxAxis.z, tMax));
        float cell_majorant[1];
        forward(deref(node).majorant_grid, int[3](cell.z, cell.y, cell.x), cell_majorant);
        float maj = cell_majorant[0] * deref(node).density_scale;
        if (maj > 0.0)
        {
            float tt = t;
            for (int safety = 0; tt < cell_exit && safety < 4096; safety++)
            {
                float dt = -log(1.0 - random()) / maj; // sample free-flight distance
                tt += dt;
                if (tt >= cell_exit)
                    break; // no interaction within this cell, move to the next one
                vec3 xp = x + wo * tt;
                float extinction_val = extinction(node, xp) * deref(node).density_scale;
                if (random() < extinction_val / maj) // accept this collision as the first hit
                {
                    _output[0] = 1.0 / (1.0 + tt); // first-hit depth - exact original formula
                    // (1/(1+tMax-d) where d is remaining distance to tMax, tMax measured from the
                    // ray's own starting position, not from box entry); tt is that same absolute
                    // distance-from-ray-origin in this DDA's own bookkeeping (x/wo are the ray's
                    // true origin/direction, unshifted by tMin - only t/tt start at tMin).
                    float sa[SPECTRAL_DIM];
                    scattering_albedo(node, xp, sa);
                    [[unroll]]
                    for (int i = 0; i < SPECTRAL_DIM; i++)
                        _output[1 + i] = sa[i]; // scattering albedo
                    _output[1 + SPECTRAL_DIM] = deref(node).g; // anisotropy
                    _output[OUT - 1] = 1.0; // S - binary hit indicator
                    return;
                }
            }
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
    // ray exited the volume with no accepted collision: _output stays all zero.
}
