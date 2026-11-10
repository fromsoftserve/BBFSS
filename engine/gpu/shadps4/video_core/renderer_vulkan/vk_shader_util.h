// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

inline constexpr u64 DirectionalFragmentShaderHash = 0xd92266b9;
inline constexpr u64 PointGridVertexShaderHash = 0x8747a367;
inline constexpr u64 DofRecombinePixelShaderHash = 0x9e1cb278;
inline constexpr u64 DofRecombineVertexShaderHash = 0xd3ca03f3;

/**
 * @brief Creates a vulkan shader module from SPIR-V bytecode.
 * @param code The SPIR-V bytecode data.
 * @param device The vulkan device handle
 */
vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device);

void CaptureSpecularModule(u64 program_hash, u64 permutation, std::span<const u32> code);
std::vector<u32> DirectionalGgxModule(u64 program_hash, bool fragment, std::span<const u32> code);

} // namespace Vulkan
