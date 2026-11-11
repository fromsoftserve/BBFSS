// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "vk_spirv_index.h"

namespace Vulkan::LocalFalloff {

struct Stats {
    uint32_t standard = 0, special = 0;
    bool Changed() const { return standard || special; }
};

inline bool Target(uint32_t hash) {
    switch (hash) {
    case 0x7cc2c044: case 0x186d69d5: case 0xd22cbfd1: case 0x3105494c:
    case 0xd75fe2ff: case 0x07d07cf3: case 0xf2531727: case 0xc6edb19a:
    case 0xb59b4c4e: case 0x505ebdbe: case 0x22b357f3: case 0x9f9ef267:
    case 0x370174f7: case 0x600f86b0:
        return true;
    default:
        return false;
    }
}

inline constexpr float DefaultCore = 0.05f;

// The light is a sphere of radius core * r rather than a point, so the curve is
//   1 / (x^2 + core^2) - 1 / (1 + core^2),  x = d / r,
// which stays finite at the light (a point source's 1/d^2 blows out whatever stands next to
// it). It is scaled to the old curve's spherical-volume energy: old (1 - x)^2 integrates to
// 4*pi/30, the new one to 4*pi * (1 - core*atan(1/core) - 1/(3*(1 + core^2))); core 0 gives the
// plain inverse square and its factor 1/20.
inline float EnergyScale(float core) {
    const float volume = core > 1e-4f ? 1.0f - core * std::atan(1.0f / core) -
                                            1.0f / (3.0f * (1.0f + core * core))
                                      : 2.0f / 3.0f;
    return 1.0f / (30.0f * volume);
}

inline Stats Patch(std::vector<uint32_t>& spv, uint32_t hash, bool enabled = true,
                   float intensity = 1.0f, float core = DefaultCore) {
    Stats stats{};
    if (!enabled || !Target(hash) || spv.size() < 5 || spv[0] != 0x07230203 || spv[3] == 0)
        return stats;

    const auto index = SpirvIndex::Build(spv, true);
    if (!index.valid) return stats;
    const auto& defs = index.defs;
    const uint32_t glsl = index.glsl, f32 = index.f32;
    const size_t first_function = index.first_function;
    if (!glsl || !f32 || index.floats != 1 || !first_function) return stats;

    const auto is_const = [&](uint32_t id, float value) { return index.IsConst(id, value); };
    const auto ext = [&](uint32_t id, uint32_t inst, uint32_t args) {
        return index.Ext(id, inst, args);
    };
    const auto binary = [&](uint32_t id, uint32_t op) { return index.Binary(id, op); };
    const auto distance_squared = [&](uint32_t id) {
        std::vector<uint32_t> terms;
        bool valid = true;
        const auto gather = [&](const auto& self, uint32_t term, uint32_t depth) -> void {
            if (!valid) return;
            if (depth > 8) {
                valid = false;
                return;
            }
            if (const auto* add = binary(term, 129)) {
                self(self, add[0], depth + 1);
                self(self, add[1], depth + 1);
            } else {
                terms.push_back(term);
            }
        };
        gather(gather, id, 0);
        if (!valid || terms.size() != 3) return false;
        for (const uint32_t term : terms) {
            const auto* square = binary(term, 133);
            if (!square || square[0] != square[1]) return false;
        }
        return true;
    };
    const auto source = [&](uint32_t id) {
        for (uint32_t depth = 0; depth != 8; ++depth) {
            const auto it = defs.find(id);
            if (it == defs.end() || it->second.op != 124) break;
            const uint32_t* w = &spv[it->second.at];
            if ((w[0] >> 16) != 4 || w[1] != f32) break;
            id = w[3];
        }
        return id;
    };

    struct Standard {
        uint32_t result, scaled_d2, zero, one;
    };
    std::map<size_t, Standard> standard;
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> diffuse;
    for (const auto& [id, def] : defs) {
        if (def.op == 133) {
            const auto* square = binary(id, 133);
            if (!square || square[0] != square[1]) continue;
            const auto* sub = binary(square[0], 131);
            if (!sub || !is_const(sub[0], 1)) continue;
            const auto* sqrt = ext(sub[1], 31, 1);
            if (!sqrt) continue;
            const auto* clamp = ext(sqrt[0], 43, 3);
            if (!clamp || !is_const(clamp[1], 0) || !is_const(clamp[2], 1)) continue;
            const auto* product = binary(clamp[0], 133);
            if (!product) continue;
            for (int side = 0; side < 2; ++side) {
                if (distance_squared(product[side]))
                    standard[def.at] = {id, clamp[0], clamp[1], clamp[2]};
            }
        } else if (def.op == 12) {
            const auto* max = ext(id, 40, 2);
            if (!max || !is_const(max[0], 0)) continue;
            const auto* sub = binary(max[1], 131);
            if (!sub) continue;
            const auto* div = binary(sub[0], 136);
            if (!div || !is_const(div[0], 1) || !distance_squared(div[1])) continue;
            diffuse[{div[1], source(sub[1])}] = id;
        }
    }

    struct Special {
        uint32_t result, scaled_d2, zero, one;
    };
    std::map<size_t, Special> special;
    for (const auto& [id, def] : defs) {
        if (def.op != 131) continue;
        const auto* sub = binary(id, 131);
        if (!sub || !is_const(sub[0], 1)) continue;
        const auto* clamp = ext(sub[1], 43, 3);
        if (!clamp || !is_const(clamp[1], 0) || !is_const(clamp[2], 1)) continue;
        const auto* product = binary(clamp[0], 133);
        if (!product) continue;
        for (int side = 0; side < 2; ++side) {
            if (!distance_squared(product[side])) continue;
            const auto inverse_radius = source(product[1 - side]);
            const auto native = diffuse.find({product[side], inverse_radius});
            if (native != diffuse.end()) special[def.at] = {id, clamp[0], clamp[1], clamp[2]};
        }
    }
    if (standard.empty() && special.empty()) return stats;

    uint32_t next = spv[3];
    core = std::isfinite(core) && core > 0.0f ? core : 0.0f;
    const uint32_t energy = next++, core_squared = next++, edge = next++;
    const uint32_t constant[] = {
        (4u << 16) | 43, f32, energy, std::bit_cast<uint32_t>(intensity * EnergyScale(core)),
        (4u << 16) | 43, f32, core_squared, std::bit_cast<uint32_t>(core * core),
        (4u << 16) | 43, f32, edge, std::bit_cast<uint32_t>(1.0f / (1.0f + core * core)),
    };
    std::vector<uint32_t> patched(spv.begin(), spv.begin() + 5);
    // result = max(0, 1 / (d^2/r^2 + core^2) - 1 / (1 + core^2)) * energy
    const auto emit = [&](uint32_t result, uint32_t scaled_d2, uint32_t zero, uint32_t one) {
        const uint32_t sum = next++, inverse = next++, difference = next++, matched = next++;
        const uint32_t instructions[] = {
            (5u << 16) | 129, f32, sum, scaled_d2, core_squared,
            (5u << 16) | 136, f32, inverse, one, sum,
            (5u << 16) | 131, f32, difference, inverse, edge,
            (7u << 16) | 12, f32, matched, glsl, 40, zero, difference,
            (5u << 16) | 133, f32, result, matched, energy,
        };
        patched.insert(patched.end(), std::begin(instructions), std::end(instructions));
    };
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16;
        if (i == first_function)
            patched.insert(patched.end(), std::begin(constant), std::end(constant));
        if (const auto site = standard.find(i); site != standard.end()) {
            const auto& s = site->second;
            emit(s.result, s.scaled_d2, s.zero, s.one);
            ++stats.standard;
        } else if (const auto site = special.find(i); site != special.end()) {
            const auto& s = site->second;
            emit(s.result, s.scaled_d2, s.zero, s.one);
            ++stats.special;
        } else {
            patched.insert(patched.end(), spv.begin() + i, spv.begin() + i + size);
        }
        i += size;
    }
    patched[3] = next;
    spv = std::move(patched);
    return stats;
}

} // namespace Vulkan::LocalFalloff
