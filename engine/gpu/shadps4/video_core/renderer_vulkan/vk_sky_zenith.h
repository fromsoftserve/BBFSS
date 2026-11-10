// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <array>
#include "video_core/buffer_cache/buffer.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {
class Instance;
class Scheduler;
class SkyZenith {
public:
    // Identifies the panorama's contents: the texture cache image and the hashes of the guest
    // memory it was last uploaded from. A source that may not be current has stable == false.
    struct SourceKey {
        vk::Image image;
        u64 uid = 0, hash = 0;
        bool stable = false;
        bool operator==(const SourceKey&) const = default;
    };
    SkyZenith(const Instance& instance, Scheduler& scheduler);
    ~SkyZenith();
    bool Enabled() const { return enabled; }
    vk::ImageView Filter(vk::DescriptorImageInfo source, const SourceKey& key,
                         vk::DescriptorBufferInfo geometry, u32 count);
private:
    const Instance& instance;
    Scheduler& scheduler;
    bool enabled = false, initialized = false, cached = false;
    SourceKey cached_key;
    vk::UniqueDescriptorSetLayout set;
    vk::UniquePipelineLayout layout;
    std::array<vk::UniquePipeline, 3> pipelines;
    vk::UniqueSampler sampler;
    VideoCore::UniqueImage horizontal, result;
    std::array<vk::UniqueImageView, 12> horizontal_views, result_views;
    vk::UniqueImageView result_view;
    std::unique_ptr<VideoCore::Buffer> validation;
};
}
