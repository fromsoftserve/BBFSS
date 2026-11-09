// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "video_core/renderer_vulkan/vk_local_falloff.h"

namespace Vulkan::LocalLightUnify {

struct Stats {
    uint32_t pairs = 0;
    bool Changed() const { return pairs != 0; }
};

inline uint32_t ExpectedPairs(uint32_t hash) {
    switch (hash) {
    case 0x370174f7: case 0x7cc2c044:
        return 1;
    case 0x186d69d5: case 0x600f86b0:
        return 2;
    case 0xd22cbfd1:
        return 3;
    case 0x3105494c:
        return 4;
    case 0xd75fe2ff:
        return 5;
    case 0x07d07cf3:
        return 6;
    case 0xf2531727:
        return 7;
    case 0xc6edb19a:
        return 8;
    case 0xb59b4c4e:
        return 9;
    case 0x505ebdbe:
        return 10;
    case 0x22b357f3:
        return 11;
    case 0x9f9ef267:
        return 12;
    default:
        return 0;
    }
}

inline Stats Patch(std::vector<uint32_t>& spv, uint32_t hash, bool enabled = true) {
    Stats stats{};
    const uint32_t expected = ExpectedPairs(hash);
    if (!enabled || !expected || !LocalFalloff::Target(hash) || spv.size() < 5 ||
        spv[0] != 0x07230203 || spv[3] == 0) {
        return stats;
    }

    struct Def {
        size_t at;
        uint32_t size, op;
    };
    std::vector<Def> instructions;
    std::unordered_map<uint32_t, Def> defs;
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_set<uint32_t> light_buffers;
    uint32_t u32 = 0, f32 = 0, boolean = 0;
    uint32_t uints = 0, floats = 0, bools = 0;
    size_t first_function = 0;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16, op = spv[i] & 0xffff;
        if (!size || size > spv.size() - i) return stats;
        const Def def{i, size, op};
        instructions.push_back(def);
        switch (op) {
        case 20:
            if (size == 2) boolean = spv[i + 1], ++bools;
            break;
        case 21:
            if (size == 4 && spv[i + 2] == 32 && spv[i + 3] == 0) {
                u32 = spv[i + 1], ++uints;
            }
            break;
        case 22:
            if (size == 3 && spv[i + 2] == 32) f32 = spv[i + 1], ++floats;
            break;
        case 43:
            if (size == 4) {
                defs[spv[i + 2]] = def;
                if (spv[i + 1] == u32) constants[spv[i + 2]] = spv[i + 3];
            }
            break;
        case 54:
            if (!first_function) first_function = i;
            break;
        case 59:
        case 61:
        case 65:
            if (size >= 4) defs[spv[i + 2]] = def;
            break;
        case 71:
            if (size == 4 && spv[i + 2] == 33 && spv[i + 3] == 1) {
                light_buffers.insert(spv[i + 1]);
            }
            break;
        case 124:
        case 128:
            if (size >= 4) defs[spv[i + 2]] = def;
            break;
        default:
            break;
        }
        i += size;
    }
    if (!u32 || !f32 || !boolean || uints != 1 || floats != 1 || bools != 1 ||
        !first_function || light_buffers.empty()) {
        return stats;
    }

    struct Group {
        std::array<uint32_t, 6> loads{};
        std::array<uint32_t, 6> chains{};
        std::array<Def, 6> load_defs{};
        std::array<Def, 6> chain_defs{};
    };
    std::map<std::pair<uint32_t, uint32_t>, Group> groups;
    for (const auto& [id, load] : defs) {
        if (load.op != 61 || load.size != 4 || spv[load.at + 1] != u32) continue;
        const auto chain_it = defs.find(spv[load.at + 3]);
        if (chain_it == defs.end()) continue;
        const Def chain = chain_it->second;
        const uint32_t* c = &spv[chain.at];
        if (chain.op != 65 || chain.size != 6 || !light_buffers.contains(c[3])) continue;
        const auto variable = defs.find(c[3]);
        if (variable == defs.end() || variable->second.op != 59 || spv[variable->second.at + 3] != 2) {
            continue;
        }
        const auto zero = constants.find(c[4]);
        if (zero == constants.end() || zero->second != 0) continue;
        const auto add_it = defs.find(c[5]);
        if (add_it == defs.end()) continue;
        const Def add = add_it->second;
        const uint32_t* a = &spv[add.at];
        if (add.op != 128 || add.size != 5 || a[1] != u32) continue;
        const auto offset = constants.find(a[3]);
        if (offset == constants.end() || offset->second % 16 < 4 || offset->second % 16 > 10 ||
            offset->second % 16 == 7) {
            continue;
        }
        const uint32_t light = offset->second / 16;
        const uint32_t component = offset->second % 16;
        const uint32_t channel = component - (component < 8 ? 4 : 5);
        auto& group = groups[{a[4], light}];
        if (group.loads[channel]) return stats;
        group.loads[channel] = id;
        group.chains[channel] = c[2];
        group.load_defs[channel] = load;
        group.chain_defs[channel] = chain;
    }
    if (groups.size() != expected) return stats;
    for (uint32_t light = 0; light < expected; ++light) {
        bool found = false;
        for (const auto& [key, group] : groups) {
            if (key.second != light) continue;
            if (found) return stats;
            found = true;
            for (int channel = 0; channel < 6; ++channel) {
                if (!group.loads[channel]) return stats;
            }
        }
        if (!found) return stats;
    }

    std::unordered_map<uint32_t, uint32_t> replacements;
    std::unordered_set<uint32_t> color_loads;
    for (const auto& [key, group] : groups) {
        for (const uint32_t load : group.loads) color_loads.insert(load);
    }
    for (const auto& instruction : instructions) {
        if (instruction.size > 2 && color_loads.contains(spv[instruction.at + 2]) &&
            (instruction.op == 61 || instruction.op == 65 || instruction.op == 128)) {
            continue;
        }
        for (uint32_t word = 1; word < instruction.size; ++word) {
            if (!color_loads.contains(spv[instruction.at + word])) continue;
            if (instruction.op != 124 || instruction.size != 4 || word != 3 ||
                spv[instruction.at + 1] != f32) {
                return stats;
            }
        }
    }

    std::map<size_t, std::vector<uint32_t>> additions;
    uint32_t next = spv[3];
    const std::array<uint32_t, 3> weights{
        std::bit_cast<uint32_t>(0.2126f), std::bit_cast<uint32_t>(0.7152f),
        std::bit_cast<uint32_t>(0.0722f)};
    const std::array<uint32_t, 3> weight_ids{next++, next++, next++};
    for (const auto& [key, group] : groups) {
        size_t insertion = 0;
        for (int channel = 0; channel < 3; ++channel) {
            const Def load = group.load_defs[channel];
            insertion = std::max(insertion, load.at + load.size);
        }
        const std::unordered_set<uint32_t> current_loads(group.loads.begin(), group.loads.end());
        for (const auto& instruction : instructions) {
            if (instruction.at >= insertion) break;
            if (instruction.size > 2 && current_loads.contains(spv[instruction.at + 2]) &&
                instruction.op == 61) {
                continue;
            }
            for (uint32_t word = 1; word < instruction.size; ++word) {
                if (current_loads.contains(spv[instruction.at + word])) return Stats{};
            }
        }

        std::vector<uint32_t> code;
        std::array<uint32_t, 6> selected;
        std::array<uint32_t, 6> values;
        for (int channel = 3; channel < 6; ++channel) {
            const Def add = defs.at(spv[group.chain_defs[channel].at + 5]);
            const uint32_t index = next++, chain = next++;
            selected[channel] = next++;
            const uint32_t* a = &spv[add.at];
            const uint32_t* c = &spv[group.chain_defs[channel].at];
            code.insert(code.end(), {a[0], a[1], index, a[3], a[4]});
            code.insert(code.end(), {c[0], c[1], chain, c[3], c[4], index});
            code.insert(code.end(), {(4u << 16) | 61, u32, selected[channel], chain});
        }
        for (int channel = 0; channel < 6; ++channel) {
            values[channel] = next++;
            const uint32_t source = channel < 3 ? group.loads[channel] : selected[channel];
            code.insert(code.end(), {(4u << 16) | 124, f32, values[channel], source});
        }
        std::array<uint32_t, 2> luminance;
        for (int side = 0; side < 2; ++side) {
            std::array<uint32_t, 3> terms;
            for (int channel = 0; channel < 3; ++channel) {
                terms[channel] = next++;
                code.insert(code.end(), {(5u << 16) | 133, f32, terms[channel],
                                         values[side * 3 + channel], weight_ids[channel]});
            }
            const uint32_t xy = next++;
            luminance[side] = next++;
            code.insert(code.end(), {(5u << 16) | 129, f32, xy, terms[0], terms[1],
                                     (5u << 16) | 129, f32, luminance[side], xy, terms[2]});
        }
        const uint32_t use_specular = next++;
        code.insert(code.end(), {(5u << 16) | 186, boolean, use_specular, luminance[1], luminance[0]});
        for (int channel = 0; channel < 3; ++channel) {
            selected[channel] = next++;
            code.insert(code.end(), {(6u << 16) | 169, u32, selected[channel], use_specular,
                                     selected[channel + 3], group.loads[channel]});
            replacements[group.loads[channel]] = selected[channel];
            replacements[group.loads[channel + 3]] = selected[channel];
        }
        additions[insertion] = std::move(code);
        ++stats.pairs;
    }

    std::vector<uint32_t> patched(spv.begin(), spv.begin() + 5);
    for (const auto& instruction : instructions) {
        if (instruction.at == first_function) {
            for (int channel = 0; channel < 3; ++channel) {
                patched.insert(patched.end(), {(4u << 16) | 43, f32, weight_ids[channel], weights[channel]});
            }
        }
        const size_t start = patched.size();
        patched.insert(patched.end(), spv.begin() + instruction.at,
                       spv.begin() + instruction.at + instruction.size);
        if (instruction.op == 124 && instruction.size == 4 && spv[instruction.at + 1] == f32) {
            if (const auto replacement = replacements.find(spv[instruction.at + 3]);
                replacement != replacements.end()) {
                patched[start + 3] = replacement->second;
            }
        }
        if (const auto addition = additions.find(instruction.at + instruction.size);
            addition != additions.end()) {
            patched.insert(patched.end(), addition->second.begin(), addition->second.end());
        }
    }
    patched[3] = next;
    spv = std::move(patched);
    return stats;
}

} // namespace Vulkan::LocalLightUnify
