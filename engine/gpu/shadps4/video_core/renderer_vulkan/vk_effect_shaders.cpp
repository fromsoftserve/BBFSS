// SPDX-License-Identifier: GPL-3.0-or-later
#include "video_core/renderer_vulkan/vk_effect_shaders.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>

#include "bbport_settings.h"
#include "video_core/host_shaders/gtao_frag.h"
#include "video_core/host_shaders/gtao_hq_frag.h"
#include "video_core/host_shaders/bloom_downsample_frag.h"
#include "video_core/host_shaders/native_motion_blur_frag.h"
#include "video_core/host_shaders/ssr_frag.h"
#include "video_core/renderer_vulkan/vk_ggx_lobes.h"
#include "video_core/renderer_vulkan/vk_ggx_patch.h"
#include "video_core/renderer_vulkan/vk_local_falloff.h"
#include "video_core/renderer_vulkan/vk_local_light_unify.h"
#include "video_core/renderer_vulkan/vk_pom_patch.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_spirv_index.h"
#include "video_core/renderer_vulkan/vk_ssr_composite_patch.h"

namespace Vulkan::EffectShaders {

namespace {
constexpr u64 WaterHash = 0xe3021062;
constexpr size_t CompositeLiftedWords = 2606;
constexpr u32 CompositeLiftedChecksum = 0xa5e815fc;

std::atomic<bool> gtao_active{false}, ssr_active{false}, composite_active{false};

bool PatchSharpness(std::vector<u32>& spv) {
    return GgxLobes::ScaleLightSharpness(spv, BbSettings::Get().startup_light_sharpness) != 0;
}

bool LobesWanted() {
    const auto& s = BbSettings::Get();
    return s.startup_local_ggx || s.startup_specular_aa || s.startup_env_ggx_tails;
}

bool SharpnessWanted() {
    return BbSettings::Get().startup_light_sharpness > 1.0f;
}

// PatchLobes then PatchSharpness; the module is scanned once unless the lobes rewrite it.
bool PatchSpecular(std::vector<u32>& spv) {
    const bool lobes_wanted = LobesWanted(), sharpness_wanted = SharpnessWanted();
    if (!lobes_wanted && !sharpness_wanted) {
        return false;
    }
    const auto& s = BbSettings::Get();
    auto index = SpirvIndex::Build(spv);
    const bool lobed =
        lobes_wanted &&
        GgxLobes::Patch(spv, {s.startup_local_ggx, s.startup_specular_aa, s.startup_env_ggx_tails},
                        index)
            .Changed();
    if (!sharpness_wanted) {
        return lobed;
    }
    if (lobed) {
        index = SpirvIndex::Build(spv);
    }
    return GgxLobes::ScaleLightSharpness(spv, s.startup_light_sharpness, index) != 0 || lobed;
}

bool Wanted(int effect) {
    return BbSettings::Get().startup_effects[effect];
}

u32 Checksum(std::span<const u32> spv) {
    u32 checksum = 2166136261u;
    for (const u32 word : spv) {
        checksum = (checksum ^ word) * 16777619u;
    }
    return checksum;
}

bool Fits(const Shader::Info& info, const Shader::Profile& profile,
          const Shader::Backend::Bindings& start, size_t buffers, size_t images, size_t samplers) {
    return info.sw_stage == Shader::SwStage::Fragment && start.unified == 0 && start.buffer == 0 &&
           profile.uniform_dword_layout && !profile.uniform_exact_base &&
           info.buffers.size() == buffers && info.images.size() == images &&
           info.samplers.size() == samplers &&
           std::ranges::all_of(info.buffers, [](const auto& buffer) { return buffer.is_uniform; });
}

void Declined(const char* name, const Shader::Info& info, const Shader::Profile& profile,
              const Shader::Backend::Bindings& start) {
    static std::atomic<u32> printed{0};
    if (printed.fetch_add(1, std::memory_order_relaxed) < 8) {
        std::printf("%s: left off, shader %08x does not have the expected layout here (start %u, "
                    "%zu buffers, %zu images, %zu samplers, dword uniforms %d)\n",
                    name, u32(info.pgm_hash), start.unified, info.buffers.size(),
                    info.images.size(), info.samplers.size(), int(profile.uniform_dword_layout));
    }
}

bool PatchWater(std::vector<u32>& spv) {
    constexpr size_t Words = 13751;
    constexpr u32 OriginalChecksum = 0xf7a3a6ee;
    constexpr u32 OffsetResultId = 1307;
    constexpr u32 OpConstant = 43, OpFMul = 133;
    if (spv.size() != Words || Checksum(spv) != OriginalChecksum) {
        return false;
    }
    size_t fmul = 0;
    for (size_t i = 5; i < spv.size(); i += spv[i] >> 16) {
        if ((spv[i] & 0xffff) == OpFMul && spv[i + 2] == OffsetResultId) {
            fmul = i;
            break;
        }
    }
    if (!fmul) {
        return false;
    }
    const u32 type_id = spv[fmul + 1];
    u32 half_id = 0;
    for (size_t i = 5; i < spv.size() && !half_id; i += spv[i] >> 16) {
        if ((spv[i] & 0xffff) == OpConstant && spv[i + 1] == type_id &&
            spv[i + 3] == std::bit_cast<u32>(0.5f)) {
            half_id = spv[i + 2];
        }
    }
    if (!half_id) {
        return false;
    }
    const u32 scaled_id = spv[3]++;
    spv[fmul + 2] = scaled_id;
    const std::array<u32, 5> scale{(5u << 16) | OpFMul, type_id, OffsetResultId, scaled_id, half_id};
    spv.insert(spv.begin() + fmul + 5, scale.begin(), scale.end());
    return true;
}
}

std::span<const u32> Replacement(const Shader::Info& info, const Shader::Profile& profile,
                                 const Shader::Backend::Bindings& start) {
    if (info.hw_stage != Shader::HwStage::Fragment) {
        return {};
    }
    if (info.pgm_hash == SsaoHash && Wanted(BbSettings::EffectGtao)) {
        if (Fits(info, profile, start, 3, 3, 3)) {
            const bool high = BbSettings::Get().startup_ao_quality == 1;
            if (!gtao_active.exchange(true, std::memory_order_relaxed)) {
                std::printf(high ? "GTAO: in use in place of the game's SSAO shader (High: three "
                                   "directions, spatial denoise, full internal resolution)\n"
                                 : "GTAO: in use in place of the game's SSAO shader\n");
            }
            return high ? std::span<const u32>(GTAO_HQ_FRAG) : std::span<const u32>(GTAO_FRAG);
        }
        Declined("GTAO", info, profile, start);
    } else if (info.pgm_hash == SsrHash && Wanted(BbSettings::EffectModernSsr)) {
        if (Fits(info, profile, start, 1, 3, 3)) {
            if (!ssr_active.exchange(true, std::memory_order_relaxed)) {
                std::printf("Improved reflections: in use\n");
            }
            return SSR_FRAG;
        }
        Declined("Improved reflections", info, profile, start);
    } else if (info.pgm_hash == MotionBlurHash &&
               BbSettings::Get().startup_motion_blur_quality == 1) {
        if (Fits(info, profile, start, 1, 2, 2)) {
            return NATIVE_MOTION_BLUR_FRAG;
        }
        Declined("Motion blur quality", info, profile, start);
    } else if (info.pgm_hash == BloomDownsampleHash &&
               BbSettings::Get().startup_bloom_quality == BbSettings::BloomQualityOriginal) {
        if (Fits(info, profile, start, 1, 1, 1)) {
            static std::atomic<bool> active{false};
            if (!active.exchange(true, std::memory_order_relaxed)) {
                std::printf("Bloom stabilization: in use\n");
            }
            return BLOOM_DOWNSAMPLE_FRAG;
        }
        Declined("Bloom stabilization", info, profile, start);
    }
    return {};
}

std::optional<std::vector<u32>> Patched(const Shader::Info& info, std::span<const u32> spv) {
    if (info.hw_stage != Shader::HwStage::Fragment) {
        return std::nullopt;
    }
    if (info.pgm_hash == SsrCompositeHash) {
        if (!Wanted(BbSettings::EffectModernSsr)) {
            return std::nullopt;
        }
        const u32 checksum = Checksum(spv);
        const bool recompiled =
            spv.size() == kSsrCompositeOriginalWords && checksum == kSsrCompositeOriginalChecksum;
        const bool lifted = spv.size() == CompositeLiftedWords && checksum == CompositeLiftedChecksum;
        if (!recompiled && !lifted) {
            static std::atomic<bool> printed{false};
            if (!printed.exchange(true)) {
                std::printf("Improved reflections: the composite shader differs on this GPU's "
                            "profile (%zu words, %08x); the game's composite stays\n",
                            spv.size(), checksum);
            }
            return std::nullopt;
        }
        composite_active.store(true, std::memory_order_relaxed);
        return std::vector<u32>(kSsrCompositeSpv.begin(), kSsrCompositeSpv.end());
    }
    if (info.pgm_hash == WaterHash && Wanted(BbSettings::EffectModernSsr)) {
        std::vector<u32> patched(spv.begin(), spv.end());
        if (PatchWater(patched)) {
            return patched;
        }
        return std::nullopt;
    }
    if (info.pgm_hash == Ggx::DirectionalHash) {
        auto ggx = DirectionalGgxModule(info.pgm_hash, true, spv);
        if (!ggx.empty()) {
            PatchSharpness(ggx);
            return ggx;
        }
    }
    if (Wanted(BbSettings::EffectPomCavity) && Pom::Targets(info.pgm_hash)) {
        std::vector<u32> patched(spv.begin(), spv.end());
        Pom::PatchPom(info.pgm_hash, true, patched);
        if (patched.size() != spv.size()) {
            LocalLightUnify::Patch(patched, info.pgm_hash,
                                   BbSettings::Get().startup_uniform_local_light);
            LocalFalloff::Patch(patched, info.pgm_hash,
                                BbSettings::Get().startup_local_inverse_square,
                                BbSettings::Get().startup_local_light_intensity);
            PatchSpecular(patched);
            return patched;
        }
    }
    const auto& settings = BbSettings::Get();
    const bool falloff_possible = LocalFalloff::Target(info.pgm_hash) &&
                                  (settings.startup_uniform_local_light ||
                                   settings.startup_local_inverse_square);
    if (!falloff_possible && !LobesWanted() && !SharpnessWanted()) {
        return std::nullopt;
    }
    std::vector<u32> lobes(spv.begin(), spv.end());
    const bool unified = LocalLightUnify::Patch(lobes, info.pgm_hash,
                                                BbSettings::Get().startup_uniform_local_light)
                             .Changed();
    const bool falloff = LocalFalloff::Patch(lobes, info.pgm_hash,
                                             BbSettings::Get().startup_local_inverse_square,
                                             BbSettings::Get().startup_local_light_intensity)
                             .Changed();
    const bool specular = PatchSpecular(lobes);
    if (specular || falloff || unified) {
        return lobes;
    }
    return std::nullopt;
}

bool KeepRecompiled(const Shader::Info& info) {
    if (info.hw_stage != Shader::HwStage::Fragment) {
        return false;
    }
    return ((info.pgm_hash == SsrCompositeHash || info.pgm_hash == WaterHash) &&
            Wanted(BbSettings::EffectModernSsr)) ||
           (Pom::Targets(info.pgm_hash) && Wanted(BbSettings::EffectPomCavity)) ||
           (info.pgm_hash == Ggx::DirectionalHash && BbSettings::Get().startup_directional_ggx);
}

bool GtaoActive() {
    return gtao_active.load(std::memory_order_relaxed);
}

bool SsrActive() {
    return ssr_active.load(std::memory_order_relaxed);
}

bool SsrCompositeActive() {
    return composite_active.load(std::memory_order_relaxed);
}

}
