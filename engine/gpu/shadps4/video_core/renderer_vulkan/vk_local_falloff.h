// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <string_view>
#include <unordered_map>
#include <vector>

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

inline Stats Patch(std::vector<uint32_t>& spv, uint32_t hash, bool enabled = true) {
    Stats stats{};
    if (!enabled || !Target(hash) || spv.size() < 5 || spv[0] != 0x07230203 || spv[3] == 0)
        return stats;

    struct Def { size_t at; uint32_t op; };
    std::unordered_map<uint32_t, Def> defs;
    std::unordered_map<uint32_t, uint32_t> constants;
    uint32_t glsl = 0, f32 = 0, floats = 0;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16, op = spv[i] & 0xffff;
        if (!size || size > spv.size() - i) return stats;
        if (op == 11 && size >= 3) {
            const char* name = reinterpret_cast<const char*>(&spv[i + 2]);
            if (std::string_view(name, strnlen(name, (size - 2) * 4)) == "GLSL.std.450")
                glsl = spv[i + 1];
        }
        if (op == 22 && size == 3 && spv[i + 2] == 32) f32 = spv[i + 1], ++floats;
        if (op == 43 && size == 4) constants[spv[i + 2]] = spv[i + 3];
        if ((op == 12 || op == 43 || op == 124 || (op >= 129 && op <= 136)) && size >= 3)
            defs[spv[i + 2]] = {i, op};
        i += size;
    }
    if (!glsl || !f32 || floats != 1) return stats;

    const auto is_const = [&](uint32_t id, float value) {
        const auto it = constants.find(id);
        return it != constants.end() && it->second == std::bit_cast<uint32_t>(value);
    };
    const auto ext = [&](uint32_t id, uint32_t inst, uint32_t args) -> const uint32_t* {
        const auto it = defs.find(id);
        if (it == defs.end() || it->second.op != 12) return nullptr;
        const uint32_t* w = &spv[it->second.at];
        if ((w[0] >> 16) != 5 + args || w[1] != f32 || w[3] != glsl || w[4] != inst) return nullptr;
        return w + 5;
    };
    const auto binary = [&](uint32_t id, uint32_t op) -> const uint32_t* {
        const auto it = defs.find(id);
        if (it == defs.end() || it->second.op != op) return nullptr;
        const uint32_t* w = &spv[it->second.at];
        if ((w[0] >> 16) != 5 || w[1] != f32) return nullptr;
        return w + 3;
    };
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
        uint32_t result, d2, inverse_radius, zero, one;
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
                    standard[def.at] = {id, product[side], product[1 - side],
                                        clamp[1], clamp[2]};
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

    std::map<size_t, std::pair<uint32_t, uint32_t>> special;
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
            const auto native = diffuse.find({product[side], source(product[1 - side])});
            if (native != diffuse.end()) special[def.at] = {id, native->second};
        }
    }
    if (standard.empty() && special.empty()) return stats;

    uint32_t next = spv[3];
    std::vector<uint32_t> patched(spv.begin(), spv.begin() + 5);
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16;
        if (const auto site = standard.find(i); site != standard.end()) {
            const auto& s = site->second;
            const uint32_t inverse = next++, difference = next++;
            const uint32_t instructions[] = {
                (5u << 16) | 136, f32, inverse, s.one, s.d2,
                (5u << 16) | 131, f32, difference, inverse, s.inverse_radius,
                (7u << 16) | 12, f32, s.result, glsl, 40, s.zero, difference,
            };
            patched.insert(patched.end(), std::begin(instructions), std::end(instructions));
            ++stats.standard;
        } else if (const auto site = special.find(i); site != special.end()) {
            const uint32_t copy[] = {(4u << 16) | 83, f32, site->second.first, site->second.second};
            patched.insert(patched.end(), std::begin(copy), std::end(copy));
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
