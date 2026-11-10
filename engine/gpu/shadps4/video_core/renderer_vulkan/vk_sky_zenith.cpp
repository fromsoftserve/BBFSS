// SPDX-License-Identifier: GPL-3.0-or-later
#include "video_core/renderer_vulkan/vk_sky_zenith.h"
#include <algorithm>
#include <cstdlib>
#include <span>
#include "video_core/host_shaders/sky_zenith_validate_comp.h"
#include "video_core/host_shaders/sky_zenith_horizontal_comp.h"
#include "video_core/host_shaders/sky_zenith_vertical_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {
namespace {
constexpr vk::ImageSubresourceRange AllMips{vk::ImageAspectFlagBits::eColor, 0, 12, 0, 1};
// verified, applied, then 12 mips of indirect dispatch arguments (see sky_zenith.comp).
constexpr u32 ValidationBytes = (2 + 12 * 3) * 4;
constexpr u32 ArgsOffset = 2 * 4;
void ComputeBarrier(vk::CommandBuffer cmd) {
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader |
                        vk::PipelineStageFlagBits2::eDrawIndirect,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite |
                         vk::AccessFlagBits2::eIndirectCommandRead};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}
}

SkyZenith::SkyZenith(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    const char* setting = std::getenv("BB_SKY_ZENITH_FIX");
    enabled = setting && setting[0] == '1' && setting[1] == '\0';
    if (!enabled) return;
    const auto device = instance.GetDevice();
    sampler = Check<"create sky zenith sampler">(device.createSamplerUnique({
        .magFilter = vk::Filter::eNearest, .minFilter = vk::Filter::eNearest,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eRepeat,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge, .maxLod = 11.0f}));
    const std::array<std::span<const u32>, 3> codes{
        SKY_ZENITH_VALIDATE_COMP, SKY_ZENITH_HORIZONTAL_COMP, SKY_ZENITH_VERTICAL_COMP};
    const std::array<vk::DescriptorSetLayoutBinding, 5> bindings{{
        {0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}}};
    set = Check<"create sky zenith descriptors">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()), .pBindings = bindings.data()}));
    const vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute, 0, 12};
    layout = Check<"create sky zenith layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1, .pSetLayouts = &*set,
        .pushConstantRangeCount = 1, .pPushConstantRanges = &push}));
    for (u32 i = 0; i < pipelines.size(); ++i) {
        const auto module = CompileSPV(codes[i], device);
        pipelines[i] = Check<"create sky zenith pipeline">(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                .layout = *layout}));
        device.destroyShaderModule(module);
    }
    validation = std::make_unique<VideoCore::Buffer>(
        instance, 0, ValidationBytes, VideoCore::MemoryType::DeviceLocal,
        "Dream panorama validation");
    horizontal = VideoCore::UniqueImage(device, instance.GetAllocator());
    result = VideoCore::UniqueImage(device, instance.GetAllocator());
    const vk::ImageCreateInfo info{
        .imageType = vk::ImageType::e2D, .format = vk::Format::eR16G16B16A16Sfloat,
        .extent = {2048, 512, 1}, .mipLevels = 12, .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1, .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
        .initialLayout = vk::ImageLayout::eUndefined};
    horizontal.Create(info);
    result.Create(info);
    for (u32 mip = 0; mip < 12; ++mip) {
        const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, mip, 1, 0, 1};
        horizontal_views[mip] = Check<"create sky zenith horizontal view">(
            device.createImageViewUnique({.image = vk::Image(horizontal),
                .viewType = vk::ImageViewType::e2D, .format = info.format, .subresourceRange = range}));
        result_views[mip] = Check<"create sky zenith result view">(
            device.createImageViewUnique({.image = vk::Image(result),
                .viewType = vk::ImageViewType::e2D, .format = info.format, .subresourceRange = range}));
    }
    result_view = Check<"create sky zenith sampled view">(device.createImageViewUnique({
        .image = vk::Image(result), .viewType = vk::ImageViewType::e2D,
        .format = info.format, .subresourceRange = AllMips}));
}
SkyZenith::~SkyZenith() = default;

// The validate pass runs on every call: the geometry buffer is rewritten on the GPU by the sky
// seam pass for each draw, so its verdict is only known there. It also decides, on the GPU,
// whether the filter passes have work: they are dispatched indirectly with zero groups unless
// the source changed (force) or the verdict differs from the one the result was filtered with.
vk::ImageView SkyZenith::Filter(vk::DescriptorImageInfo source, const SourceKey& key,
                              vk::DescriptorBufferInfo geometry, u32 count) {
    if (!enabled || !source.imageView || !geometry.buffer) return {};
    source.sampler = *sampler;
    const u32 force = cached && key.stable && key == cached_key ? 0u : 1u;
    cached = key.stable;
    cached_key = key;
    scheduler.EndRendering();
    if (!initialized) {
        initialized = true;
        scheduler.Record([h = vk::Image(horizontal), r = vk::Image(result)](vk::CommandBuffer cmd) {
            std::array<vk::ImageMemoryBarrier2, 2> barriers{};
            for (u32 i = 0; i < 2; ++i)
                barriers[i] = {.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                    .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral,
                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                    .image = i ? r : h, .subresourceRange = AllMips};
            cmd.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = barriers.data()});
        });
    }
    for (u32 step = 0; step < 25; ++step) {
        const u32 pass_index = step == 0 ? 0 : (step % 2 ? 1 : 2);
        const u32 mip = step == 0 ? 0 : (step - 1) / 2;
        const std::array<vk::DescriptorImageInfo, 3> images{source,
            vk::DescriptorImageInfo{{}, *horizontal_views[mip], vk::ImageLayout::eGeneral},
            vk::DescriptorImageInfo{{}, *result_views[mip], vk::ImageLayout::eGeneral}};
        const std::array<vk::DescriptorBufferInfo, 2> buffers{
            geometry, vk::DescriptorBufferInfo{validation->Handle(), 0, ValidationBytes}};
        struct Params { u32 mip, count, force; };
        const Params params{mip, count, force};
        scheduler.Record([images, buffers, params, pass_index, args = validation->Handle(),
                          pipe = *pipelines[pass_index], layout = *layout](vk::CommandBuffer cmd) {
            ComputeBarrier(cmd);
            std::array<vk::WriteDescriptorSet, 5> writes{};
            for (u32 i = 0; i < 5; ++i) {
                writes[i] = {.dstBinding = i, .descriptorCount = 1,
                    .descriptorType = i == 0 ? vk::DescriptorType::eCombinedImageSampler :
                                      i < 3 ? vk::DescriptorType::eStorageImage :
                                              vk::DescriptorType::eStorageBuffer,
                    .pImageInfo = i < 3 ? &images[i] : nullptr,
                    .pBufferInfo = i >= 3 ? &buffers[i - 3] : nullptr};
            }
            cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipe);
            cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmd.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(params), &params);
            if (pass_index == 0) {
                cmd.dispatch(1, 1, 1);
            } else {
                cmd.dispatchIndirect(args, ArgsOffset + params.mip * 12);
            }
        });
    }
    scheduler.Record([](vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 after{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &after});
    });
    return *result_view;
}
}
