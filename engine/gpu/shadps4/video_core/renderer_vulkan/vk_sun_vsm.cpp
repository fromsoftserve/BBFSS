// SPDX-License-Identifier: GPL-3.0-or-later

#include "video_core/renderer_vulkan/vk_sun_vsm.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>

#include "video_core/host_shaders/sun_vsm_comp.h"
#include "video_core/host_shaders/sun_vsm_draw_vert.h"
#include "video_core/renderer_vulkan/vk_effect_pass.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {
namespace {
constexpr u32 Physical = 4096;
constexpr u32 TableEntries = 16 * 32 * 32;
constexpr u32 PhysicalPages = 1024;
constexpr u32 MaxDraws = 8192;
// Per frame slot: the casters, then the changed casters (each caster plus each removed one).
constexpr u32 SlotDraws = 3 * MaxDraws;
constexpr u32 ChangedFirst = MaxDraws;
constexpr u32 MaxItems = 12u << 20;
constexpr u32 Slots = 4;
// Bounds slots outlive one frame's casters only by the removed ones, so two per draw suffice.
constexpr u32 BoundsSlots = 2 * MaxDraws;
constexpr u32 BoundsWords = 12;
// sun_vsm.glsl: args, clear args, eight counters, eight words, six vec4, 16 cascades of 32
// bytes, 16 bounds, 24 bytes per table entry and 12 per physical page.
constexpr vk::DeviceSize StateBytes =
    16 + 16 + 32 + 32 + 6 * 16 + 16 * 32 + 16 * 16 + 24 * TableEntries + 12 * PhysicalPages;
constexpr vk::DeviceSize CounterOffset = 32;
constexpr vk::DeviceSize CounterBytes = 64;
constexpr vk::DeviceSize ClearArgsOffset = 16;
constexpr u32 ResetBit = 1u << 8;
constexpr u32 CasterDynamic = 1, CasterFresh = 2, CasterRemoved = 4;

enum Mode : u32 {
    ModeSetup = 0,
    ModeMark,
    ModeInvalidate,
    ModeFreeList,
    ModeAllocate,
    ModeBoundsReset,
    ModeBounds,
    ModeDirty,
    ModeResolve,
    ModeBin,
    ModeFinalize,
    ModeSample,
};

struct Push {
    u32 mode;
    u32 buf_off;
    std::array<float, 2> size;
    std::array<float, 12> inv_view;
    std::array<float, 12> view;
    u32 draw_count;
    u32 triangle_count;
    u32 max_items;
    u32 frame_index;
};
static_assert(sizeof(Push) == 128);

struct Draw {
    u64 vertices;
    u64 indices;
    u32 first;
    u32 count;
    u32 slot;
    u32 flags;
};
static_assert(sizeof(Draw) == 32);

constexpr auto Compute = vk::PipelineStageFlagBits2::eComputeShader;
constexpr auto DepthTests = vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                            vk::PipelineStageFlagBits2::eLateFragmentTests;
}

SunVsm::SunVsm(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    ready = Setup();
    std::printf("Sun virtual shadow map: %s\n",
                ready ? "ready, 16 cascades of 4096^2, 1024 pages of 128^2 in a 4096^2 atlas"
                      : "off - setup failed");
}

SunVsm::~SunVsm() {
    scheduler.Finish();
    const auto device = instance.GetDevice();
    if (physical_view) {
        device.destroyImageView(physical_view);
    }
    if (physical) {
        vmaDestroyImage(instance.GetAllocator(), physical, physical_allocation);
    }
    for (Buffer* b : {&state, &items, &draws, &readback, &bounds, &bounds_readback}) {
        Free(*b);
    }
}

bool SunVsm::Make(Buffer& out, vk::DeviceSize size, bool host, VkBufferUsageFlags usage) {
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
    };
    VmaAllocationCreateInfo alloc_ci{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (host) {
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    }
    VkBuffer raw{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &raw, &out.allocation,
                        &info) != VK_SUCCESS) {
        return false;
    }
    out.buffer = raw;
    out.mapped = info.pMappedData;
    return true;
}

void SunVsm::Free(Buffer& buffer) {
    if (buffer.buffer) {
        vmaDestroyBuffer(instance.GetAllocator(), buffer.buffer, buffer.allocation);
    }
    buffer = {};
}

bool SunVsm::Setup() {
    const auto device = instance.GetDevice();
    const auto features = instance.GetPhysicalDevice().getFeatures();
    if (!features.shaderClipDistance) {
        std::printf("Sun virtual shadow map: off - no clip distances\n");
        return false;
    }
    const auto depth_features =
        instance.GetPhysicalDevice().getFormatProperties(vk::Format::eD32Sfloat).optimalTilingFeatures;
    if (!(depth_features & vk::FormatFeatureFlagBits::eDepthStencilAttachment) ||
        !(depth_features & vk::FormatFeatureFlagBits::eSampledImage)) {
        return false;
    }
    const vk::DeviceSize item_bytes = vk::DeviceSize(MaxItems) * 8;
    if (!EffectPass::HasRoom(instance, item_bytes + vk::DeviceSize(Physical) * Physical * 4)) {
        std::printf("Sun virtual shadow map: off - not enough video memory\n");
        return false;
    }
    using enum vk::DescriptorType;
    if (!EffectPass::CreateSampler(instance, sampler, vk::Filter::eNearest) ||
        !EffectPass::Create(instance, compute_set, compute_layout, compute_pipeline,
                            {eStorageBuffer, eCombinedImageSampler, eStorageBuffer, eStorageBuffer,
                             eStorageBuffer, eCombinedImageSampler, eStorageImage, eStorageBuffer},
                            sizeof(Push), SUN_VSM_COMP, "Sun virtual shadow map")) {
        return false;
    }
    if (!Make(state, StateBytes, false,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
        !Make(items, item_bytes, false, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !Make(draws, vk::DeviceSize(Slots) * SlotDraws * sizeof(Draw), true,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        !Make(readback, Slots * CounterBytes, true, VK_BUFFER_USAGE_TRANSFER_DST_BIT) ||
        !Make(bounds, vk::DeviceSize(BoundsSlots) * BoundsWords * 4, false,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT) ||
        !Make(bounds_readback, vk::DeviceSize(Slots) * BoundsSlots * BoundsWords * 4, true,
              VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        return false;
    }
    std::memset(readback.mapped, 0, Slots * CounterBytes);
    free_slots.reserve(BoundsSlots);
    for (u32 s = BoundsSlots; s-- > 0;) {
        free_slots.push_back(s);
    }

    const VkImageCreateInfo image_ci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_D32_SFLOAT,
        .extent = {Physical, Physical, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo image_alloc{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VkImage raw{};
    if (vmaCreateImage(instance.GetAllocator(), &image_ci, &image_alloc, &raw, &physical_allocation,
                       nullptr) != VK_SUCCESS) {
        return false;
    }
    physical = raw;
    const auto [view_result, view] = device.createImageView({
        .image = physical,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eD32Sfloat,
        .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
    });
    if (view_result != vk::Result::eSuccess) {
        return false;
    }
    physical_view = view;

    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings{{
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex},
    }};
    auto set = device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    });
    if (set.result != vk::Result::eSuccess) {
        return false;
    }
    draw_set = std::move(set.value);
    auto layout = device.createPipelineLayoutUnique({.setLayoutCount = 1, .pSetLayouts = &*draw_set});
    if (layout.result != vk::Result::eSuccess) {
        return false;
    }
    draw_layout = std::move(layout.value);
    const std::span<const u32> code = SUN_VSM_DRAW_VERT;
    const auto [module_result, module] =
        device.createShaderModule({.codeSize = code.size_bytes(), .pCode = code.data()});
    if (module_result != vk::Result::eSuccess) {
        return false;
    }
    SetObjectName(device, module, "Sun virtual shadow map draw");
    const vk::PipelineShaderStageCreateInfo stage{
        .stage = vk::ShaderStageFlagBits::eVertex, .module = module, .pName = "main"};
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo assembly{
        .topology = vk::PrimitiveTopology::eTriangleList};
    const vk::Viewport viewport{0.0f, 0.0f, float(Physical), float(Physical), 0.0f, 1.0f};
    const vk::Rect2D scissor{{0, 0}, {Physical, Physical}};
    const vk::PipelineViewportStateCreateInfo viewport_state{
        .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};
    const vk::PipelineRasterizationStateCreateInfo raster{
        .depthClampEnable = features.depthClamp,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eNone,
        .lineWidth = 1.0f,
    };
    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1};
    const vk::PipelineDepthStencilStateCreateInfo depth{
        .depthTestEnable = true,
        .depthWriteEnable = true,
        .depthCompareOp = vk::CompareOp::eLess,
    };
    const vk::PipelineColorBlendStateCreateInfo blend{};
    const vk::PipelineRenderingCreateInfo rendering{.depthAttachmentFormat = vk::Format::eD32Sfloat};
    vk::GraphicsPipelineCreateInfo pipeline_ci{
        .pNext = &rendering,
        .stageCount = 1,
        .pStages = &stage,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &raster,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depth,
        .pColorBlendState = &blend,
        .layout = *draw_layout,
    };
    auto pipeline = device.createGraphicsPipelineUnique({}, pipeline_ci);
    // Pages drawn again are first reset to the far plane, whatever they held.
    const vk::PipelineDepthStencilStateCreateInfo clear_depth{
        .depthTestEnable = true,
        .depthWriteEnable = true,
        .depthCompareOp = vk::CompareOp::eAlways,
    };
    pipeline_ci.pDepthStencilState = &clear_depth;
    auto clear = device.createGraphicsPipelineUnique({}, pipeline_ci);
    device.destroyShaderModule(module);
    if (pipeline.result != vk::Result::eSuccess || clear.result != vk::Result::eSuccess) {
        return false;
    }
    draw_pipeline = std::move(pipeline.value);
    clear_pipeline = std::move(clear.value);
    return true;
}

bool SunVsm::TakeSlot(u32& slot) {
    if (free_slots.empty()) {
        return false;
    }
    slot = free_slots.back();
    free_slots.pop_back();
    return true;
}

bool SunVsm::Record(std::span<const Caster> casters, const Inputs& in) {
    ++stats.calls;
    if (!ready || casters.empty() || !in.raw_view || !in.depth) {
        ++stats.no_casters;
        return false;
    }
    const u32 slot = u32(frame % Slots);
    if (recorded_frame != frame) {
        if (slot_ticks[slot]) {
            scheduler.Wait(slot_ticks[slot]);
            slot_ticks[slot] = 0;
            const auto* counters =
                static_cast<const u32*>(readback.mapped) + slot * (CounterBytes / 4);
            stats.items += counters[0];
            stats.pages += counters[1];
            stats.item_overflow += counters[2];
            stats.page_overflow += counters[3];
            stats.drawn += counters[4];
            stats.requested += counters[5];
            stats.moved += counters[13];
            stats.marked += counters[14];
            stats.marked_max = std::max<u64>(stats.marked_max, counters[15]);
            ++stats.samples;
            const auto* words = static_cast<const float*>(bounds_readback.mapped) +
                                vk::DeviceSize(slot) * BoundsSlots * BoundsWords;
            for (const auto& [address, bounds_slot] : slot_casters[slot]) {
                const auto it = tracked.find(address);
                if (it == tracked.end() || it->second.slot != bounds_slot) {
                    continue;
                }
                std::array<float, 6> box;
                std::memcpy(box.data(), words + bounds_slot * BoundsWords + 6, sizeof(box));
                if (box[0] <= box[3]) {
                    boxes[address] = box;
                } else {
                    boxes.erase(address);
                }
            }
        }
        slot_casters[slot].clear();
        recorded_frame = frame;
    } else {
        // A second sun pass in one frame would overwrite this slot's draw table in flight.
        ++stats.repeats;
        return false;
    }
    auto* table = static_cast<Draw*>(draws.mapped) + slot * SlotDraws;
    auto* changed = table + ChangedFirst;
    // The cached pages are dropped when the map missed a frame: what moved meanwhile is unknown.
    const bool reset = !cleared || drawn_frame + 1 != frame;
    if (reset) {
        tracked.clear();
        boxes.clear();
        for (auto& list : slot_casters) {
            list.clear();
        }
        free_slots.clear();
        for (u32 s = BoundsSlots; s-- > 0;) {
            free_slots.push_back(s);
        }
        previous.clear();
    }
    current.clear();
    u32 count = 0, triangles = 0, changed_count = 0, changed_triangles = 0;
    for (const Caster& caster : casters) {
        if (count == MaxDraws || !caster.vertices || !caster.indices || !caster.triangles) {
            continue;
        }
        u32 flags = caster.dynamic ? CasterDynamic : 0;
        auto it = tracked.find(caster.vertices);
        if (it == tracked.end()) {
            u32 bounds_slot;
            if (!TakeSlot(bounds_slot)) {
                continue;
            }
            it = tracked.emplace(caster.vertices, Tracked{bounds_slot, 0}).first;
            flags |= CasterFresh;
        } else if (it->second.seen == frame) {
            continue;
        }
        it->second.seen = frame;
        current.push_back(caster.vertices);
        const Draw draw{caster.vertices, caster.indices, triangles, caster.triangles,
                        it->second.slot, flags};
        table[count++] = draw;
        triangles += caster.triangles;
        if (flags != 0 || caster.changed) {
            slot_casters[slot].emplace_back(caster.vertices, it->second.slot);
            changed[changed_count] = draw;
            changed[changed_count].first = changed_triangles;
            ++changed_count;
            changed_triangles += caster.triangles;
        }
    }
    if (!count) {
        ++stats.no_casters;
        return false;
    }
    // Casters gone since the last frame clear their old pages, then give their slot back.
    u32 removed = 0;
    for (const u64 address : previous) {
        const auto it = tracked.find(address);
        if (it == tracked.end() || it->second.seen == frame) {
            continue;
        }
        changed[changed_count++] = {address, 0, changed_triangles, 0, it->second.slot,
                                    CasterRemoved};
        free_slots.push_back(it->second.slot);
        tracked.erase(it);
        boxes.erase(address);
        ++removed;
    }
    std::swap(previous, current);
    drawn_frame = frame;
    stats.casters += count;
    stats.triangles += triangles;
    stats.changed += changed_count - removed;
    stats.removed += removed;
    stats.resets += reset ? 1 : 0;
    ++stats.frames;
    slot_ticks[slot] = scheduler.CurrentTick();

    Push push{
        .mode = ModeSetup,
        .buf_off = in.constants_offset_word,
        .size = {float(in.width), float(in.height)},
        .inv_view = in.inv_view,
        .view = in.view,
        .draw_count = count,
        .triangle_count = triangles,
        .max_items = MaxItems,
        .frame_index = in.frame_index,
    };
    const bool clear = !cleared;
    cleared = true;
    const vk::DeviceSize draws_offset = vk::DeviceSize(slot) * SlotDraws * sizeof(Draw);
    scheduler.EndRendering();
    // Recorded as five stages, each its own profiler label.
    auto body = 
        [in, push, clear, reset, slot, draws_offset, changed_count, changed_triangles,
         state = state.buffer, items = items.buffer, draws = draws.buffer,
         readback = readback.buffer, bounds = bounds.buffer,
         bounds_readback = bounds_readback.buffer, physical = physical,
         physical_view = physical_view, sampler = *sampler, compute_pipeline = *compute_pipeline,
         compute_layout = *compute_layout, draw_pipeline = *draw_pipeline,
         clear_pipeline = *clear_pipeline,
         draw_layout = *draw_layout](vk::CommandBuffer cmdbuf, int stage) mutable {
            const auto memory = [&](vk::PipelineStageFlags2 src, vk::AccessFlags2 src_access,
                                    vk::PipelineStageFlags2 dst, vk::AccessFlags2 dst_access) {
                const vk::MemoryBarrier2 barrier{.srcStageMask = src, .srcAccessMask = src_access,
                                                 .dstStageMask = dst, .dstAccessMask = dst_access};
                cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
            };
            constexpr auto ReadWrite =
                vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
            if (stage == 0) {
            if (clear) {
                cmdbuf.fillBuffer(state, 0, VK_WHOLE_SIZE, 0);
            }
            memory(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eMemoryWrite |
                                                                 vk::AccessFlagBits2::eMemoryRead,
                   Compute | vk::PipelineStageFlagBits2::eDrawIndirect |
                       vk::PipelineStageFlagBits2::eVertexShader,
                   ReadWrite | vk::AccessFlagBits2::eIndirectCommandRead);
            // The atlas keeps its pages between frames; its contents are undefined only at first.
            const std::array<vk::ImageMemoryBarrier2, 2> to_general{{
                {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                 .dstStageMask = Compute,
                 .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
                 .oldLayout = vk::ImageLayout::eUndefined,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = in.raw_image,
                 .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}},
                {.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                 .srcAccessMask = vk::AccessFlagBits2::eMemoryRead,
                 .dstStageMask = DepthTests | Compute,
                 .dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite |
                                  vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                 .oldLayout = clear ? vk::ImageLayout::eUndefined : vk::ImageLayout::eGeneral,
                 .newLayout = vk::ImageLayout::eGeneral,
                 .image = physical,
                 .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}},
            }};
            cmdbuf.pipelineBarrier2(
                {.imageMemoryBarrierCount = u32(to_general.size()), .pImageMemoryBarriers = to_general.data()});
            }

            const vk::DescriptorBufferInfo constants{in.constants, in.constants_offset,
                                                     in.constants_range};
            const vk::DescriptorImageInfo depth{sampler, in.depth, vk::ImageLayout::eGeneral};
            const vk::DescriptorBufferInfo state_info{state, 0, VK_WHOLE_SIZE};
            const vk::DescriptorBufferInfo items_info{items, 0, VK_WHOLE_SIZE};
            const vk::DescriptorBufferInfo draws_info{draws, draws_offset,
                                                      vk::DeviceSize(SlotDraws) * sizeof(Draw)};
            const vk::DescriptorImageInfo physical_info{sampler, physical_view,
                                                        vk::ImageLayout::eGeneral};
            const vk::DescriptorImageInfo raw_info{{}, in.raw_view, vk::ImageLayout::eGeneral};
            const vk::DescriptorBufferInfo bounds_info{bounds, 0, VK_WHOLE_SIZE};
            const std::array<vk::WriteDescriptorSet, 8> writes{{
                {.dstBinding = 0, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &constants},
                {.dstBinding = 1, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &depth},
                {.dstBinding = 2, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &state_info},
                {.dstBinding = 3, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &items_info},
                {.dstBinding = 4, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &draws_info},
                {.dstBinding = 5, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                 .pImageInfo = &physical_info},
                {.dstBinding = 6, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &raw_info},
                {.dstBinding = 7, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &bounds_info},
            }};
            const auto bind_compute = [&] {
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, compute_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, compute_layout, 0,
                                            writes);
            };
            const auto run = [&](u32 mode, u32 x, u32 y) {
                push.mode = mode;
                cmdbuf.pushConstants(compute_layout, vk::ShaderStageFlagBits::eCompute, 0,
                                     sizeof(push), &push);
                cmdbuf.dispatch(x, y, 1);
                memory(Compute, vk::AccessFlagBits2::eShaderWrite, Compute, ReadWrite);
            };
            const auto run_linear = [&](u32 mode, u32 threads) {
                const u32 groups = std::max((threads + 63) / 64, 1u);
                const u32 x = std::min(groups, 32768u);
                run(mode, x, (groups + x - 1) / x);
            };
            const u32 width_groups = (in.width + 7) / 8, height_groups = (in.height + 7) / 8;
            if (stage == 0) {
                bind_compute();
                run(ModeSetup | (reset ? ResetBit : 0u), (TableEntries + 63) / 64, 1);
                run(ModeMark, width_groups, height_groups);
                run_linear(ModeInvalidate, TableEntries);
                run_linear(ModeFreeList, PhysicalPages);
                run_linear(ModeAllocate, TableEntries);
                return;
            }
            if (stage == 1) {
                if (changed_count) {
                    bind_compute();
                    const Push main = push;
                    push.draw_count = changed_count;
                    push.triangle_count = changed_triangles;
                    push.max_items = ChangedFirst;
                    run_linear(ModeBoundsReset, changed_count);
                    run_linear(ModeBounds, changed_triangles);
                    run_linear(ModeDirty, changed_count);
                    push = main;
                    // The kept bounds go back to the host, which drops kept casters the game
                    // would have drawn.
                    const vk::DeviceSize bytes = vk::DeviceSize(BoundsSlots) * BoundsWords * 4;
                    memory(Compute, vk::AccessFlagBits2::eShaderWrite,
                           vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead);
                    cmdbuf.copyBuffer(bounds, bounds_readback,
                                      vk::BufferCopy{0, vk::DeviceSize(slot) * bytes, bytes});
                    memory(vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferRead,
                           Compute, vk::AccessFlagBits2::eShaderWrite);
                }
                return;
            }
            if (stage == 2) {
            bind_compute();
            run_linear(ModeResolve, TableEntries);
            run_linear(ModeBin, push.triangle_count);
            run(ModeFinalize, 1, 1);
            memory(Compute, vk::AccessFlagBits2::eShaderWrite,
                   vk::PipelineStageFlagBits2::eDrawIndirect | vk::PipelineStageFlagBits2::eVertexShader |
                       vk::PipelineStageFlagBits2::eCopy,
                   vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eShaderRead |
                       vk::AccessFlagBits2::eTransferRead);
            cmdbuf.copyBuffer(state, readback,
                              vk::BufferCopy{CounterOffset, vk::DeviceSize(slot) * CounterBytes,
                                             CounterBytes});
            return;
            }
            if (stage == 3) {

            const vk::RenderingAttachmentInfo depth_attachment{
                .imageView = physical_view,
                .imageLayout = vk::ImageLayout::eGeneral,
                .loadOp = clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
                .storeOp = vk::AttachmentStoreOp::eStore,
                .clearValue = vk::ClearValue{.depthStencil = {1.0f, 0}},
            };
            cmdbuf.beginRendering({
                .renderArea = {{0, 0}, {Physical, Physical}},
                .layerCount = 1,
                .pDepthAttachment = &depth_attachment,
            });
            const std::array<vk::WriteDescriptorSet, 3> draw_writes{{
                {.dstBinding = 0, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &state_info},
                {.dstBinding = 1, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &items_info},
                {.dstBinding = 2, .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageBuffer, .pBufferInfo = &draws_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, clear_pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, draw_layout, 0,
                                        draw_writes);
            cmdbuf.drawIndirect(state, ClearArgsOffset, 1, 16);
            // The clear and the casters' depth writes are ordered within the one pass.
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, draw_pipeline);
            cmdbuf.drawIndirect(state, 0, 1, 16);
            cmdbuf.endRendering();
            const vk::ImageMemoryBarrier2 to_sample{
                .srcStageMask = DepthTests,
                .srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                .dstStageMask = Compute,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = physical,
                .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
            };
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_sample});
            return;
            }
            bind_compute();
            run(ModeSample, width_groups, height_groups);
        };
    const auto shared = std::make_shared<decltype(body)>(std::move(body));
    const auto record_stage = [&](u64 key, const char* name, int stage) {
        EffectPass::Mark(key, name);
        scheduler.RecordCrumb({.name = "Sun virtual shadow map"},
                              [shared, stage](vk::CommandBuffer cmdbuf) { (*shared)(cmdbuf, stage); });
    };
    record_stage(15, "sun virtual shadow map", 0);
    record_stage(17, "sun VSM caster bounds", 1);
    record_stage(18, "sun VSM page binning", 2);
    record_stage(19, "sun VSM page draws", 3);
    record_stage(20, "sun VSM sampling", 4);
    // The guest pipelines' dynamic state must be emitted again after the page draw.
    scheduler.GetDynamicState().Invalidate();
    return true;
}

const std::array<float, 6>* SunVsm::Box(u64 vertices) const {
    const auto it = boxes.find(vertices);
    return it == boxes.end() ? nullptr : &it->second;
}

void SunVsm::OnFrameEnd() {
    if (!ready) {
        return;
    }
    ++frame;
    if (frame % 600 == 0) {
        std::printf("Sun virtual shadow map: last 600 frames %llu sun passes, %llu drawn, %llu "
                    "without casters, %llu repeats\n",
                    (unsigned long long)stats.calls, (unsigned long long)stats.frames,
                    (unsigned long long)stats.no_casters, (unsigned long long)stats.repeats);
        if (!stats.frames) {
            stats = {};
        }
    }
    if (stats.frames >= 600) {
        const double f = double(stats.frames);
        const double s = double(std::max<u64>(stats.samples, 1));
        std::printf("Sun virtual shadow map: %.0f casters (%.1f changed, %.2f removed), %.0f "
                    "triangles, %.0f page draws; %.0f pages needed, %.1f new, %.1f drawn of 1024; "
                    "dropped %.1f page draws, %.1f pages; %llu resets; %.1f casters moved marking "
                    "%.0f pages (most by one %llu)\n",
                    double(stats.casters) / f, double(stats.changed) / f,
                    double(stats.removed) / f, double(stats.triangles) / f,
                    double(stats.items) / s, double(stats.requested) / s, double(stats.pages) / s,
                    double(stats.drawn) / s, double(stats.item_overflow) / s,
                    double(stats.page_overflow) / s, (unsigned long long)stats.resets,
                    double(stats.moved) / s, double(stats.marked) / s,
                    (unsigned long long)stats.marked_max);
        stats = {};
    }
}

}
