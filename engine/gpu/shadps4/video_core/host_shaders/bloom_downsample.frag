// SPDX-License-Identifier: GPL-2.0-or-later
#version 450
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_samplerless_texture_functions : require

layout(scalar, set = 0, binding = 0) uniform Constants { uint data[16384]; } constants;
layout(set = 0, binding = 1) uniform texture2D color_image;
layout(set = 0, binding = 2) uniform sampler color_sampler;
layout(push_constant) uniform Push { layout(offset = 80) uint buffer_offset; } pc;
layout(location = 0) in vec4 uv;
layout(location = 0) out vec4 color;

float constant(int index) {
    return uintBitsToFloat(constants.data[index + ((pc.buffer_offset & 255u) >> 2u)]);
}

float weight(vec3 rgb) {
    return 1.0 / (1.0 + dot(rgb, vec3(0.2126, 0.7152, 0.0722)));
}

void main() {
    const ivec2 size = textureSize(sampler2D(color_image, color_sampler), 0);
    const ivec2 base = ivec2(gl_FragCoord.xy) * 2;
    const vec4 a = texelFetch(color_image, min(base + ivec2(0, 0), size - 1), 0);
    const vec4 b = texelFetch(color_image, min(base + ivec2(1, 0), size - 1), 0);
    const vec4 c = texelFetch(color_image, min(base + ivec2(0, 1), size - 1), 0);
    const vec4 d = texelFetch(color_image, min(base + ivec2(1, 1), size - 1), 0);
    const vec4 weights = vec4(weight(a.rgb), weight(b.rgb), weight(c.rgb), weight(d.rgb));
    const vec3 rgb = (a.rgb * weights.x + b.rgb * weights.y + c.rgb * weights.z +
                      d.rgb * weights.w) / dot(weights, vec4(1.0));
    const float alpha = dot(vec4(a.a, b.a, c.a, d.a), vec4(0.25));
    color = vec4(rgb, alpha) * vec4(constant(616), constant(617), constant(618), constant(619));
}
