// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
class Scheduler;

/// bbport: a virtual shadow map for the sun (prototype). Sixteen clipmap cascades of 4096^2
/// virtual texels in 128^2 pages; only pages the depth buffer needs are given space in one 4096^2
/// physical depth atlas. The casters are the ray tracing capture's world space shadow pass
/// triangles (vk_rt_shadow.cpp), binned to pages and drawn page-clipped in one indirect depth-only
/// draw. Pages are cached between frames: only new pages and those under casters whose world
/// bounds changed (or skinned casters) are drawn again, plus a slow staggered refresh. The result
/// is the lit fraction in the ray traced sun's raw mask, so the existing filter and native-map
/// combine follow unchanged.
class SunVsm {
public:
    SunVsm(const Instance& instance, Scheduler& scheduler);
    ~SunVsm();

    [[nodiscard]] bool Ready() const noexcept {
        return ready;
    }

    struct Caster {
        u64 vertices = 0;
        u64 indices = 0;
        u32 triangles = 0;
        /// Skinned: its pages are drawn again every frame it is seen.
        bool dynamic = false;
        /// World copy written since the last call: its bounds are checked for movement.
        bool changed = false;
    };
    struct Inputs {
        vk::Buffer constants;
        vk::DeviceSize constants_offset = 0;
        vk::DeviceSize constants_range = 0;
        vk::ImageView depth;
        vk::Image raw_image;
        vk::ImageView raw_view;
        std::array<float, 12> view{};
        std::array<float, 12> inv_view{};
        u32 constants_offset_word = 0;
        u32 frame_index = 0;
        u32 width = 0;
        u32 height = 0;
    };
    /// Records every pass; leaves `raw_view` (R8, General) holding the lit fraction.
    bool Record(std::span<const Caster> casters, const Inputs& inputs);

    void OnFrameEnd();

    /// World bounds (min xyz, max xyz) of a caster's world copy as last measured on the GPU (a
    /// few frames late), or null when not measured yet.
    [[nodiscard]] const std::array<float, 6>* Box(u64 vertices) const;

private:
    struct Buffer {
        vk::Buffer buffer;
        VmaAllocation allocation{};
        void* mapped = nullptr;
    };
    bool Setup();
    bool Make(Buffer& out, vk::DeviceSize size, bool host, VkBufferUsageFlags usage);
    void Free(Buffer& buffer);
    bool TakeSlot(u32& slot);

    const Instance& instance;
    Scheduler& scheduler;
    bool ready = false;
    bool cleared = false;

    vk::UniqueSampler sampler;
    vk::UniqueDescriptorSetLayout compute_set, draw_set;
    vk::UniquePipelineLayout compute_layout, draw_layout;
    vk::UniquePipeline compute_pipeline, draw_pipeline, clear_pipeline;
    Buffer state, items, draws, readback, bounds, bounds_readback;
    vk::Image physical;
    VmaAllocation physical_allocation{};
    vk::ImageView physical_view;

    std::array<u64, 4> slot_ticks{};
    u64 frame = 0;
    u64 recorded_frame = ~0ull;
    u64 drawn_frame = ~0ull;

    // Bounds slot of each caster (by world copy address) and the frame it was last drawn.
    struct Tracked {
        u32 slot = 0;
        u64 seen = 0;
    };
    std::unordered_map<u64, Tracked> tracked;
    std::vector<u32> free_slots;
    std::vector<u64> previous, current;
    // Each frame slot's casters (world copy address, bounds slot), for the bounds readback.
    std::array<std::vector<std::pair<u64, u32>>, 4> slot_casters;
    std::unordered_map<u64, std::array<float, 6>> boxes;

    struct Stats {
        u64 frames = 0, casters = 0, triangles = 0, items = 0, pages = 0, item_overflow = 0,
            page_overflow = 0, samples = 0, calls = 0, repeats = 0, no_casters = 0, drawn = 0,
            requested = 0, resets = 0, changed = 0, removed = 0, moved = 0, marked = 0,
            marked_max = 0;
    } stats;
};

}
