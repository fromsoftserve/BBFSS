// SPDX-License-Identifier: GPL-3.0-or-later
#include "video_core/renderer_vulkan/vk_froyok_bloom.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "bbport_settings.h"
#include "common/assert.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/native/froyok_composite.h"
#include "video_core/host_shaders/froyok_bloom_downsample_comp.h"
#include "video_core/host_shaders/froyok_bloom_first_comp.h"
#include "video_core/host_shaders/froyok_bloom_upsample_comp.h"
#include "video_core/renderer_vulkan/vk_effect_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_runtime.h"
#include "video_core/renderer_vulkan/vk_scene_resolution.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace Vulkan {
namespace {
constexpr auto Compute = vk::PipelineStageFlagBits2::eComputeShader;
constexpr vk::ImageSubresourceRange ColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

void Barrier(vk::CommandBuffer cmd) {
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}
}

FroyokBloom::FroyokBloom(const Instance& instance_, Scheduler& scheduler_,
                         VideoCore::TextureCache& texture_cache_, Runtime& runtime_,
                         SceneTargets& scene_targets_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, runtime{runtime_},
      scene_targets{scene_targets_} {}

FroyokBloom::~FroyokBloom() = default;

void FroyokBloom::OnFrameEnd() {
    ready = false;
}

bool FroyokBloom::Ready() {
    if (tried) {
        return !failed;
    }
    tried = true;
    if (!EffectPass::StorageFormat(instance, vk::Format::eR16G16B16A16Sfloat) ||
        !EffectPass::CreateSampler(instance, upsample_sampler, vk::Filter::eLinear)) {
        failed = true;
        return false;
    }
    const auto device = instance.GetDevice();
    auto border = device.createSamplerUnique({
        .magFilter = vk::Filter::eLinear, .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToBorder,
        .addressModeV = vk::SamplerAddressMode::eClampToBorder,
        .addressModeW = vk::SamplerAddressMode::eClampToBorder,
        .maxLod = 0.0f, .borderColor = vk::BorderColor::eFloatTransparentBlack});
    if (border.result != vk::Result::eSuccess) {
        failed = true;
        return false;
    }
    downsample_sampler = std::move(border.value);
    using enum vk::DescriptorType;
    const std::array codes{std::span<const u32>{FROYOK_BLOOM_DOWNSAMPLE_COMP},
                           std::span<const u32>{FROYOK_BLOOM_FIRST_COMP},
                           std::span<const u32>{FROYOK_BLOOM_UPSAMPLE_COMP}};
    const std::array names{"Froyok bloom downsample", "Froyok bloom first downsample",
                           "Froyok bloom upsample"};
    for (u32 stage = 0; stage < pipelines.size(); ++stage) {
        if (!EffectPass::Create(instance, set_layout, layout, pipelines[stage],
                                {eCombinedImageSampler, eCombinedImageSampler, eStorageImage},
                                sizeof(Push), codes[stage], names[stage])) {
            failed = true;
            return false;
        }
    }
    return true;
}

FroyokBloom::Source FroyokBloom::Access(VideoCore::ImageId id) {
    auto& image = texture_cache.GetImage(id);
    if (image.info.pixel_format != vk::Format::eR16G16B16A16Sfloat ||
        image.info.resources.levels != 1 || image.info.resources.layers != 1) {
        return {};
    }
    if (scene_targets.Reduced() && scene_targets.EligibleScene(image)) {
        VideoCore::ImageViewInfo info;
        info.format = image.info.pixel_format;
        const auto proxy = scene_targets.Read(id, info);
        return {proxy.view, scene_targets.Size().width, scene_targets.Size().height};
    }
    const auto device = instance.GetDevice();
    const auto view = Check<"create Froyok HDR view">(device.createImageView({
        .image = image.GetImage(), .viewType = vk::ImageViewType::e2D,
        .format = image.info.pixel_format, .subresourceRange = ColorRange}));
    scheduler.DeferOperation([device, view] { device.destroyImageView(view); });
    runtime.Transit(&image, vk::ImageLayout::eGeneral, Compute, vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    return {view, image.info.size.width, image.info.size.height};
}

bool FroyokBloom::EnsureImages(u32 w, u32 h) {
    if (width == w && height == h && !downsampled.empty() && !upsampled.empty()) {
        return true;
    }
    const u32 levels = std::max(1, int(std::floor(std::log2(std::min(w, h)))) - 3);
    u64 bytes = 0;
    u32 x = w, y = h;
    for (u32 level = 0; level < levels; ++level) {
        x = std::max(1u, x / 2);
        y = std::max(1u, y / 2);
        bytes += 16ull * x * y;
    }
    if (!EffectPass::HasRoom(instance, bytes)) {
        return false;
    }
    for (auto* images : {&downsampled, &upsampled}) {
        for (auto& image : *images) {
            scheduler.DeferOperation(
                [old = std::move(image.image), view = std::move(image.view)]() mutable {
                    view.reset();
                    old.Destroy();
                });
        }
        images->clear();
        images->resize(levels);
    }
    width = w;
    height = h;
    const auto device = instance.GetDevice();
    for (auto* images : {&downsampled, &upsampled}) {
        for (u32 level = 0; level < levels; ++level) {
            Image& image = (*images)[level];
            image.width = std::max(1u, (w >> level) / 2);
            image.height = std::max(1u, (h >> level) / 2);
            image.image = VideoCore::UniqueImage(device, instance.GetAllocator());
            image.image.Create({
                .imageType = vk::ImageType::e2D, .format = vk::Format::eR16G16B16A16Sfloat,
                .extent = {image.width, image.height, 1}, .mipLevels = 1, .arrayLayers = 1,
                .samples = vk::SampleCountFlagBits::e1, .tiling = vk::ImageTiling::eOptimal,
                .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                .initialLayout = vk::ImageLayout::eUndefined});
            image.view = Check<"create Froyok working view">(device.createImageViewUnique({
                .image = vk::Image(image.image), .viewType = vk::ImageViewType::e2D,
                .format = vk::Format::eR16G16B16A16Sfloat, .subresourceRange = ColorRange}));
            scheduler.Record([target = vk::Image(image.image)](vk::CommandBuffer cmd) {
                const vk::ImageMemoryBarrier2 barrier{
                    .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
                    .dstStageMask = Compute, .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                    .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = target, .subresourceRange = ColorRange};
                cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
            });
        }
    }
    return true;
}

void FroyokBloom::Dispatch(u32 stage, const Push& push, vk::ImageView current,
                           vk::ImageView previous, const Image& output) {
    const vk::DescriptorImageInfo in{stage == 2 ? vk::Sampler(*upsample_sampler)
                                                : vk::Sampler(*downsample_sampler),
                                     current, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo before{*upsample_sampler, previous, vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo out{{}, *output.view, vk::ImageLayout::eGeneral};
    const u32 groups_x = (output.width + 7) / 8, groups_y = (output.height + 7) / 8;
    scheduler.Record([push, groups_x, groups_y, in, before, out, pipeline = *pipelines[stage],
                      layout = *layout](vk::CommandBuffer cmd) {
        Barrier(cmd);
        const std::array descriptors{
            vk::WriteDescriptorSet{.dstBinding = 0, .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &in},
            vk::WriteDescriptorSet{.dstBinding = 1, .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                   .pImageInfo = &before},
            vk::WriteDescriptorSet{.dstBinding = 2, .descriptorCount = 1,
                                   .descriptorType = vk::DescriptorType::eStorageImage,
                                   .pImageInfo = &out}};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, descriptors);
        cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        cmd.dispatch(groups_x, groups_y, 1);
        Barrier(cmd);
    });
}

void FroyokBloom::Generate(VideoCore::ImageId id) {
    ready = false;
    if (!id || BbSettings::Get().startup_bloom_quality != BbSettings::BloomQualityFroyok ||
        !Ready()) {
        return;
    }
    scheduler.EndRendering();
    const Source source = Access(id);
    if (!source.view || !source.width || !source.height || !EnsureImages(source.width, source.height)) {
        if (!reported) {
            reported = true;
            std::printf("Froyok bloom: unavailable for this HDR target; retaining Original Bloom\n");
        }
        return;
    }
    Push push{.source_size = {float(source.width), float(source.height)},
              .destination_size = {downsampled[0].width, downsampled[0].height},
              .pixel_size = {1.0f / source.width, 1.0f / source.height}, .blend = 0.0f, .unused = 0};
    Dispatch(1, push, source.view, source.view, downsampled[0]);
    for (size_t level = 1; level < downsampled.size(); ++level) {
        const Image& previous = downsampled[level - 1];
        const Image& output = downsampled[level];
        push = {.source_size = {float(previous.width), float(previous.height)},
                .destination_size = {output.width, output.height},
                .pixel_size = {1.0f / previous.width, 1.0f / previous.height}, .blend = 0.0f, .unused = 0};
        Dispatch(0, push, *previous.view, *previous.view, output);
    }
    for (size_t level = downsampled.size() - 1; level-- > 0;) {
        const Image& current = downsampled[level];
        const Image& previous = level + 1 == downsampled.size() - 1 ? downsampled.back()
                                                                      : upsampled[level + 1];
        const Image& output = upsampled[level];
        push = {.source_size = {float(previous.width), float(previous.height)},
                .destination_size = {output.width, output.height},
                .pixel_size = {1.0f / output.width, 1.0f / output.height},
                .blend = std::clamp(BbSettings::Get().graphics_froyok_bloom_spread.load(
                                         std::memory_order_relaxed), BbSettings::FroyokSpreadMin,
                                     BbSettings::FroyokSpreadDefault), .unused = 0};
        Dispatch(2, push, *current.view, *previous.view, output);
    }
    ready = true;
}

float FroyokBloom::Composite(const Shader::Info& fragment,
                             std::span<vk::DescriptorImageInfo> image_infos) {
    if (!ready || fragment.pgm_hash != Shader::Native::FroyokCompositeHash ||
        !fragment.uses_froyok_bloom || fragment.images.size() != 4 || image_infos.size() != 4) {
        return 0.0f;
    }
    ready = false;
    image_infos[2].imageView = *upsampled[0].view;
    image_infos[2].imageLayout = vk::ImageLayout::eGeneral;
    return std::clamp(BbSettings::Get().graphics_froyok_bloom_blend.load(std::memory_order_relaxed),
                      BbSettings::FroyokBlendMin, BbSettings::FroyokBlendMax);
}

} // namespace Vulkan
