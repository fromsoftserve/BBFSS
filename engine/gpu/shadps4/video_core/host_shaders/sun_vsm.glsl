// SPDX-License-Identifier: GPL-3.0-or-later
// Shared layout of the sun virtual shadow map (vk_sun_vsm.cpp): 16 clipmap cascades of 4096^2
// virtual texels in 128^2 pages, cascade c covering 2^(c+1) m around the camera, backed by one
// 4096^2 physical depth atlas (1024 pages).
//
// Pages are cached between frames. The page table is indexed by world page coordinates wrapped
// to the 32x32 window (a clipmap), so a page keeps its index and contents while the camera moves;
// `tag` holds the world page a table entry stands for. A page is drawn again only when it is
// newly given space, when a caster's world bounds change over it, or by a slow staggered refresh.

#define VSM_CASCADES 16
#define VSM_PAGES 32
#define VSM_PAGE 128.0
#define VSM_VIRTUAL 4096.0
#define VSM_PHYSICAL_PAGES 1024u
#define VSM_TABLE (VSM_CASCADES * VSM_PAGES * VSM_PAGES)
#define VSM_Z_RANGE 400.0
// Vertex index of the first page-clearing corner in the page draw; below it are caster triangles.
#define VSM_CLEAR_FIRST 0x40000000u

// Page flags: contents out of date (drawn when next needed), drawn this frame.
#define VSM_DIRTY 1u
#define VSM_RENDER 2u

// Caster flags (VsmDraw.flags): skinned, new to the bounds slot, gone since last frame.
#define VSM_CASTER_DYNAMIC 1u
#define VSM_CASTER_FRESH 2u
#define VSM_CASTER_REMOVED 4u

struct VsmCascade {
    vec2 origin;
    float texel;
    float page;
    ivec2 base;
    ivec2 pad;
};

struct VsmDraw {
    uvec2 vertices;
    uvec2 indices;
    uint first;
    uint count;
    uint slot;
    uint flags;
};

#define VSM_STATE_BODY                                                                          \
    uint args[4];                                                                               \
    uint clear_args[4];                                                                         \
    uint item_count;                                                                            \
    uint page_count;                                                                            \
    uint spare_counters[2];                                                                     \
    uint render_count;                                                                          \
    uint spare_count;                                                                           \
    uint reset;                                                                                 \
    uint frame;                                                                                 \
    uint free_front;                                                                            \
    uint free_back;                                                                             \
    float anchor;                                                                               \
    float height;                                                                               \
    uint anchor_set;                                                                            \
    uint spare[3];                                                                              \
    vec4 right;                                                                                 \
    vec4 up;                                                                                    \
    vec4 light;                                                                                 \
    vec4 cam;                                                                                   \
    vec4 kept_light;                                                                            \
    vec4 at;                                                                                    \
    VsmCascade cascades[VSM_CASCADES];                                                          \
    uvec4 bounds[VSM_CASCADES];                                                                 \
    uint request[VSM_TABLE];                                                                    \
    uint table[VSM_TABLE];                                                                      \
    ivec2 tag[VSM_TABLE];                                                                       \
    uint stamp[VSM_TABLE];                                                                      \
    uint flags[VSM_TABLE];                                                                      \
    uint owner[VSM_PHYSICAL_PAGES];                                                             \
    uint free_list[VSM_PHYSICAL_PAGES];                                                         \
    uint render_list[VSM_PHYSICAL_PAGES];

// Depth of a height above the cached anchor (st.anchor); the anchor only moves on a reset.
float VsmDepth(float height) {
    return 0.5 - height / (2.0 * VSM_Z_RANGE);
}

// Window page (0..31 from the clipmap's low corner) of a wrapped table page, and back.
ivec2 VsmWindow(ivec2 wrapped, ivec2 base) {
    return (wrapped - base) & (VSM_PAGES - 1);
}

uint VsmIndex(int c, ivec2 window, ivec2 base) {
    const ivec2 w = (base + window) & (VSM_PAGES - 1);
    return uint((c * VSM_PAGES + w.y) * VSM_PAGES + w.x);
}

ivec2 VsmWrapped(uint index) {
    return ivec2(index % uint(VSM_PAGES), (index / uint(VSM_PAGES)) % uint(VSM_PAGES));
}
