// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <span>
#include <vector>

#include "video_core/texture_cache/image.h"

namespace Shader {
struct Info;
}
namespace VideoCore {
class TextureCache;
}
namespace Vulkan {

class Instance;
class Runtime;
class Scheduler;
class SceneTargets;

class FroyokBloom {
public:
    FroyokBloom(const Instance& instance, Scheduler& scheduler,
                VideoCore::TextureCache& texture_cache, Runtime& runtime,
                SceneTargets& scene_targets);
    ~FroyokBloom();

    void OnFrameEnd();
    void Generate(VideoCore::ImageId source);
    float Composite(const Shader::Info& fragment,
                    std::span<vk::DescriptorImageInfo> image_infos);

private:
    struct Image {
        VideoCore::UniqueImage image;
        vk::UniqueImageView view;
        u32 width{}, height{};
    };
    struct Source {
        vk::ImageView view{};
        u32 width{}, height{};
    };
    struct Push {
        std::array<float, 2> source_size;
        std::array<u32, 2> destination_size;
        std::array<float, 2> pixel_size;
        float blend;
        u32 unused;
    };
    static_assert(sizeof(Push) == 32);

    bool Ready();
    Source Access(VideoCore::ImageId id);
    bool EnsureImages(u32 width, u32 height);
    void Dispatch(u32 stage, const Push& push, vk::ImageView current,
                  vk::ImageView previous, const Image& output);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    SceneTargets& scene_targets;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout layout;
    std::array<vk::UniquePipeline, 3> pipelines;
    vk::UniqueSampler downsample_sampler, upsample_sampler;
    std::vector<Image> downsampled, upsampled;
    u32 width{}, height{};
    bool tried{}, failed{}, ready{}, reported{};
};

} // namespace Vulkan
