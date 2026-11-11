// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/hash.h"
#include "common/logging/log.h"
#include "bbport_settings.h"
#include "video_core/renderer_vulkan/vk_ggx_patch.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace Vulkan {

std::vector<u32> DirectionalGgxModule(u64 program_hash, bool fragment, std::span<const u32> code) {
    const auto model = BbSettings::Get().startup_specular_model;
    if (!fragment || program_hash != Ggx::DirectionalHash || model == BbSettings::SpecularOriginal)
        return {};
    std::vector<u32> selected(code.begin(), code.end());
    const auto strict = model == BbSettings::SpecularGgxFull;
    const auto result = Ggx::Patch(program_hash, fragment, true, selected, strict);
    if (result != Ggx::Result::Applied) {
        LOG_ERROR(Render_Vulkan,
                  "Directional GGX {:#x}: unrecognized shader structure ({} words); "
                  "native specular retained", program_hash, code.size());
        return {};
    }
    LOG_INFO(Render_Vulkan, "Directional GGX {:#x}: {} specular enabled", program_hash,
             strict ? "strict" : "native-anchored");
    return selected;
}

vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device) {
    const vk::ShaderModuleCreateInfo shader_info = {
        .codeSize = code.size() * sizeof(u32),
        .pCode = code.data(),
    };

    auto [module_result, module] = device.createShaderModule(shader_info);
    ASSERT_MSG(module_result == vk::Result::eSuccess, "Failed to compile SPIR-V shader: {}",
               vk::to_string(module_result));
    return module;
}

void CaptureSpecularModule(u64 program_hash, u64 permutation, std::span<const u32> code) {
    static const bool enabled = [] {
        const char* flag = std::getenv("BB_SPECULAR_CAPTURE");
        return flag && flag[0] == '1' && flag[1] == '\0';
    }();
    if (!enabled || (program_hash != PointGridVertexShaderHash &&
                     program_hash != DirectionalFragmentShaderHash)) return;
    const char* directory = std::getenv("BB_CAPTURE_DIR");
    if (!directory || !directory[0]) {
        std::printf("Specular capture: BB_CAPTURE_DIR is required for compiled modules\n");
        return;
    }
    static std::mutex mutex;
    static std::unordered_set<u64> captured;
    const u64 key = HashCombine(program_hash, permutation);
    std::scoped_lock lock{mutex};
    if (captured.contains(key)) return;
    std::ostringstream name;
    name << "compiled-" << std::hex << key << ".spv";
    const auto path = std::filesystem::path(directory) / name.str();
    std::error_code error;
    if (std::filesystem::exists(path, error) || error) {
        std::printf("Specular capture: refusing to overwrite compiled module %s\n",
                    path.string().c_str());
        return;
    }
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(code.data()), std::streamsize(code.size_bytes()));
    file.close();
    if (!file) {
        std::printf("Specular capture: compiled module write failed for %s\n", path.string().c_str());
        return;
    }
    captured.insert(key);
}

} // namespace Vulkan
