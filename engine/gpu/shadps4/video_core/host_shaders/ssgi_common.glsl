// SPDX-License-Identifier: GPL-2.0-or-later

// Shared by the ssgi shaders; each declares the Constants push block (proj, jitter, size) first.

float viewZ(float depth) {
    return proj.w / min(depth - proj.z, -1e-8);
}

vec3 viewPos(vec2 pixel, float z) {
    const vec2 ndc = (pixel - jitter) / size * 2.0 - 1.0;
    return vec3(ndc.x * z / proj.x, ndc.y * z / proj.y, z);
}
