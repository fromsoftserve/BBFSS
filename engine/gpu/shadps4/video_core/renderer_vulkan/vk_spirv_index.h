// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>
#include <unordered_map>

namespace Vulkan::SpirvIndex {

struct Def {
    size_t at;
    uint32_t op;
};

// One pass over a module's instructions. `valid` is false for a short, mis-magicked or
// malformed module; the other fields are then incomplete. The index points into the words it
// was built from and is stale once those words change.
struct Index {
    const uint32_t* words = nullptr;
    size_t count = 0;
    bool valid = false;
    std::unordered_map<uint32_t, Def> defs;
    std::unordered_map<uint32_t, uint32_t> constants, vectors;
    uint32_t glsl = 0, f32 = 0, entry = 0, floats = 0;
    uint32_t sole_f32 = 0; // the float32 type id when exactly one exists, else ~0u (0 if none)
    size_t first_function = 0;

    bool IsConst(uint32_t id, float value) const {
        const auto it = constants.find(id);
        return it != constants.end() && it->second == std::bit_cast<uint32_t>(value);
    }
    // Operands of `id = ExtInst f32 glsl inst ...` taking `args` arguments, else null.
    const uint32_t* Ext(uint32_t id, uint32_t inst, uint32_t args) const {
        const auto it = defs.find(id);
        if (it == defs.end() || it->second.op != 12) return nullptr;
        const uint32_t* w = words + it->second.at;
        if ((w[0] >> 16) != 5 + args || w[1] != f32 || w[3] != glsl || w[4] != inst) return nullptr;
        return w + 5;
    }
    // Operands of the f32 binary instruction `id = op a b`, else null.
    const uint32_t* Binary(uint32_t id, uint32_t op) const {
        const auto it = defs.find(id);
        if (it == defs.end() || it->second.op != op) return nullptr;
        const uint32_t* w = words + it->second.at;
        if ((w[0] >> 16) != 5 || w[1] != f32) return nullptr;
        return w + 3;
    }
};

// Definitions recorded: ExtInst, Constant, FAdd..FDiv and, with `conversions`, FConvert.
inline Index Build(std::span<const uint32_t> spv, bool conversions = false) {
    Index x;
    x.words = spv.data();
    x.count = spv.size();
    if (spv.size() < 5 || spv[0] != 0x07230203) return x;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16, op = spv[i] & 0xffff;
        if (!size || size > spv.size() - i) return x;
        if (op == 11 && size >= 3) {
            const char* name = reinterpret_cast<const char*>(&spv[i + 2]);
            if (std::string_view(name, strnlen(name, (size - 2) * 4)) == "GLSL.std.450")
                x.glsl = spv[i + 1];
        }
        if (op == 22 && size == 3 && spv[i + 2] == 32) {
            x.f32 = spv[i + 1], ++x.floats;
            x.sole_f32 = x.sole_f32 ? ~0u : spv[i + 1];
        }
        if (op == 23 && size == 4 && spv[i + 3] == 4) x.vectors[spv[i + 1]] = spv[i + 2];
        if (op == 15 && size >= 3 && spv[i + 1] == 4 && !x.entry) x.entry = spv[i + 2];
        if (op == 43 && size == 4) x.constants[spv[i + 2]] = spv[i + 3];
        if ((op == 12 || op == 43 || (conversions && op == 124) || (op >= 129 && op <= 136)) &&
            size >= 3)
            x.defs[spv[i + 2]] = {i, op};
        if (op == 54 && !x.first_function) x.first_function = i;
        i += size;
    }
    x.valid = true;
    return x;
}

} // namespace Vulkan::SpirvIndex
