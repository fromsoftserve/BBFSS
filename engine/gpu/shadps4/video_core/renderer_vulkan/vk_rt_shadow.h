// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_sun_vsm.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
class CameraMotion;
class Instance;
class ObjectMotion;
class Scheduler;
class SunVsm;

class RtShadow {
public:
    RtShadow(const Instance& instance, Scheduler& scheduler, CameraMotion& camera_motion,
             const ObjectMotion& object_motion);
    ~RtShadow();

    [[nodiscard]] bool Capturing() const noexcept {
        return capturing;
    }
    [[nodiscard]] bool Wanted() const;
    [[nodiscard]] bool Complete() const;

    struct Draw {
        u64 shader = 0;
        u64 geometry = 0;
        u64 indices = 0;
        u64 palette = 0;
        const float* constants = nullptr;
        u32 index_size = 0;
        u32 index_count = 0;
        u32 first_vertex = 0;
        u32 first_instance = 0;
        u32 instances = 1;
        bool restart = false;
        bool strip = false;
        bool triangles = false;
        bool shadow_pass = false;
        bool camera_pass = false;
        bool solid = false;
        bool skinned = false;
        bool alpha = false;
    };
    u32 PrepareDraw(const Draw& draw);
    void BindAddresses(vk::PipelineLayout layout, u32 mode);

    bool PrepareScene();
    [[nodiscard]] vk::AccelerationStructureKHR Scene() const noexcept {
        return tlas;
    }

    struct Inputs {
        vk::Buffer constants;
        vk::DeviceSize constants_offset = 0;
        vk::DeviceSize constants_range = 0;
        vk::Buffer uniforms;
        vk::DeviceSize uniforms_offset = 0;
        vk::DeviceSize uniforms_range = 0;
        vk::ImageView depth;
        vk::ImageView target;
        u32 constants_offset_word = 0;
        u32 frame_index = 0;
        u32 width = 0;
        u32 height = 0;
    };
    bool Record(const Inputs& inputs);

    void RecordBuilds();

    void OnFrameEnd();

private:
    struct Buffer {
        vk::Buffer buffer;
        VmaAllocation allocation{};
        void* mapped = nullptr;
        u64 address = 0;
    };
    struct Entry {
        Buffer storage;
        Buffer indices;
        Buffer world;
        vk::AccelerationStructureKHR handle;
        u64 address = 0;
        u64 palette = 0;
        u32 first_vertex = 0;
        u32 vertices = 0;
        u32 triangles = 0;
        u32 instances = 1;
        u32 first_instance = 0;
        u64 scratch = 0;
        u64 update_scratch = 0;
        u64 seen_frame = 0;
        u64 built_frame = 0;
        u64 bytes = 0;
        u8 mask = 1;
        bool built = false;
        bool dynamic = false;
        bool rebuilt = false;
        bool copied = false;
        // The BLAS holds this entry's geometry (false while the scene is not being built).
        bool blas = false;
        // Value of `copies` when the world copy was last written.
        u64 copied_at = 0;
        // World bounds measured by the sun VSM (min xyz, max xyz), when has_box.
        std::array<float, 6> box{};
        bool has_box = false;
        // The sun VSM's culling result for `cull_box` under the camera signature `cull_serial`.
        std::array<float, 6> cull_box{};
        u64 cull_serial = 0;
        bool culled = false;
    };
    struct Build {
        Entry* entry;
        u32 store;
        bool update;
        std::array<float, 12> to_world;
    };

    bool Setup(const ObjectMotion& object_motion);
    bool Make(Buffer& out, vk::DeviceSize size, bool host, VkBufferUsageFlags usage);
    void Free(Buffer& buffer);
    void Destroy(Entry& entry);
    bool ToWorld(const float* constants, bool perspective, std::array<float, 12>& out) const;
    bool MakeRaw(u32 width, u32 height);
    bool PrepareCapture();
    // Whether anything this frame traces the scene. Off (sun VSM only), captures only feed the
    // VSM's world copies and no acceleration structures are built.
    [[nodiscard]] bool SceneWanted() const;
    u32 Param(u32 store, const Entry& entry);
    std::unique_ptr<Entry> MakeEntry(const Draw& draw, u8 mask);

    const Instance& instance;
    Scheduler& scheduler;
    CameraMotion& camera_motion;

    bool capturing = false;
    bool tried = false;
    bool failed = false;

    vk::UniqueDescriptorSetLayout address_layout;
    vk::UniqueDescriptorPool address_pool;
    vk::DescriptorSet capture_set;
    vk::DescriptorSet both_set;
    vk::UniqueDescriptorSetLayout trace_set, filter_set, unproject_set;
    vk::UniquePipelineLayout trace_layout, filter_layout, unproject_layout;
    vk::UniquePipeline trace_pipeline, filter_pipeline, unproject_pipeline;
    vk::UniqueSampler sampler;
    std::unique_ptr<SunVsm> vsm;
    bool vsm_tried = false;
    // SceneWanted() at the start of the frame, kept for the whole frame.
    bool need_scene = true;
    // World copies recorded so far, and the count at the last VSM pass (its changed casters).
    u64 copies = 0;
    u64 vsm_copies = 0;
    // Reused each frame: SunVsm::Record reads the casters synchronously.
    std::vector<SunVsm::Caster> vsm_casters;
    // What the culling of the kept casters depends on besides their boxes, as of the last frame
    // it ran; `cull_serial` changes whenever it differs (0: none yet).
    std::vector<std::array<float, 12>> cull_volumes;
    std::array<float, 12> cull_view{}, cull_inv_view{};
    std::array<float, 2> cull_proj{};
    std::size_t cull_drawn = 0;
    u64 cull_serial = 0;

    Buffer capture_addresses, both_addresses, params, positions, scratch, instance_data,
        tlas_storage, tlas_scratch;
    vk::AccelerationStructureKHR tlas;
    vk::Image raw_image;
    VmaAllocation raw_allocation{};
    vk::ImageView raw_view;
    u32 raw_width = 0, raw_height = 0;

    std::unordered_map<u64, std::unique_ptr<Entry>> entries;
    std::unordered_set<u64> shared_keys;
    std::unordered_map<u64, u32> occurrences;
    std::vector<const float*> frame_cameras;
    std::vector<std::array<float, 12>> frame_volumes;
    std::vector<Entry*> frame_entries;
    std::vector<Build> builds;
    std::array<u64, 4> slot_ticks{};
    u64 frame = 1;
    u32 params_used = 0;
    u32 positions_used = 0;
    u32 new_triangles = 0;
    u32 rebuilt_triangles = 0;
    u64 scratch_used = 0;
    u64 entry_bytes = 0;
    u32 missed = 0;
    u32 tlas_count = 0;
    u32 rays = 2;

    struct Stats {
        u64 draws = 0, camera_draws = 0, not_ortho = 0, no_camera = 0, alpha = 0, unsupported = 0,
            over_budget = 0, instanced = 0, dynamic_draws = 0, strips = 0, frames = 0, instances = 0,
            triangles = 0, shared_builds = 0, missed = 0, incomplete = 0, vsm_kept = 0,
            vsm_culled = 0;
    } stats;
};

}
