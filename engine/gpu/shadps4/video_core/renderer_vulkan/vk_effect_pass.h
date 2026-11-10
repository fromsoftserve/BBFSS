// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <initializer_list>
#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace Vulkan::EffectPass {

bool Create(const Instance& instance, vk::UniqueDescriptorSetLayout& set_layout,
            vk::UniquePipelineLayout& layout, vk::UniquePipeline& pipeline,
            std::initializer_list<vk::DescriptorType> types, u32 push_size,
            std::span<const u32> code, const char* name, bool dispatch_base = false);

bool CreateSampler(const Instance& instance, vk::UniqueSampler& sampler, vk::Filter filter);

bool StorageFormat(const Instance& instance, vk::Format format);

bool HasRoom(const Instance& instance, vk::DeviceSize bytes);

void Mark(u64 key, const char* name);

inline void MemoryBarrier(vk::CommandBuffer cmd, vk::PipelineStageFlags2 src_stage,
                          vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stage,
                          vk::AccessFlags2 dst_access) {
    const vk::MemoryBarrier2 barrier{.srcStageMask = src_stage, .srcAccessMask = src_access,
                                     .dstStageMask = dst_stage, .dstAccessMask = dst_access};
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
}

inline void FullBarrier(vk::CommandBuffer cmd) {
    constexpr auto All = vk::PipelineStageFlagBits2::eAllCommands;
    constexpr auto Access = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    MemoryBarrier(cmd, All, Access, All, Access);
}

inline void ComputeBarrier(vk::CommandBuffer cmd) {
    constexpr auto Compute = vk::PipelineStageFlagBits2::eComputeShader;
    MemoryBarrier(cmd, Compute, vk::AccessFlagBits2::eShaderWrite, Compute,
                  vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
}

inline void InitGeneral(vk::CommandBuffer cmd, vk::Image image,
                        const vk::ImageSubresourceRange& range) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image, .subresourceRange = range};
    cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

}
