// SPDX-License-Identifier: GPL-3.0-or-later
// Sun virtual shadow map page draw (vk_sun_vsm.cpp): each binned item is one caster triangle in
// one resident page. It is placed straight into that page of the physical atlas and clipped to
// the page edges, so a depth-only draw fills every page at once.

#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require

#include "sun_vsm.glsl"

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Indices { uint i[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer Vertices { vec4 p[]; };

layout(scalar, set = 0, binding = 0) readonly buffer State { VSM_STATE_BODY } st;
layout(scalar, set = 0, binding = 1) readonly buffer Items { uvec2 items[]; };
layout(scalar, set = 0, binding = 2) readonly buffer Draws { VsmDraw draws[]; };

out gl_PerVertex {
    vec4 gl_Position;
    float gl_ClipDistance[4];
};

const uvec2 ClearCorners[6] = uvec2[6](uvec2(0, 0), uvec2(1, 0), uvec2(0, 1), uvec2(0, 1),
                                        uvec2(1, 0), uvec2(1, 1));

void main() {
    if (uint(gl_VertexIndex) >= VSM_CLEAR_FIRST) {
        // A page drawn this frame is reset to the far plane first (compare always pipeline).
        const uint v = uint(gl_VertexIndex) - VSM_CLEAR_FIRST;
        const uint slot = st.render_list[v / 6u];
        const vec2 physical = (vec2(slot % uint(VSM_PAGES), slot / uint(VSM_PAGES)) +
                               vec2(ClearCorners[v % 6u])) * VSM_PAGE;
        gl_Position = vec4(physical / VSM_VIRTUAL * 2.0 - 1.0, 1.0, 1.0);
        gl_ClipDistance[0] = 1.0;
        gl_ClipDistance[1] = 1.0;
        gl_ClipDistance[2] = 1.0;
        gl_ClipDistance[3] = 1.0;
        return;
    }
    const uvec2 item = items[gl_VertexIndex / 3];
    const uint corner = uint(gl_VertexIndex) % 3u;
    const VsmDraw draw = draws[item.y >> 14];
    const uint page = item.y & 0x3fffu;
    const vec3 rel = Vertices(draw.vertices).p[Indices(draw.indices).i[item.x * 3u + corner]].xyz -
                     st.cam.xyz;
    const int c = int(page / uint(VSM_PAGES * VSM_PAGES));
    const VsmCascade cascade = st.cascades[c];
    const vec2 texel =
        (vec2(dot(rel, st.right.xyz), dot(rel, st.up.xyz)) - cascade.origin) / cascade.texel;
    const vec2 page_xy = vec2(VsmWindow(VsmWrapped(page), cascade.base));
    const vec2 local = texel - page_xy * VSM_PAGE;
    const uint slot = st.table[page] - 1u;
    const vec2 physical =
        vec2(slot % uint(VSM_PAGES), slot / uint(VSM_PAGES)) * VSM_PAGE + local;
    gl_Position = vec4(physical / VSM_VIRTUAL * 2.0 - 1.0,
                       VsmDepth(dot(rel, st.light.xyz) + st.height), 1.0);
    gl_ClipDistance[0] = local.x;
    gl_ClipDistance[1] = VSM_PAGE - local.x;
    gl_ClipDistance[2] = local.y;
    gl_ClipDistance[3] = VSM_PAGE - local.y;
}
