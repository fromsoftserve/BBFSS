// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <unordered_map>
#include <vector>

#include "vk_spirv_index.h"

namespace Vulkan::Ggx {

constexpr uint64_t DirectionalHash = 0xd92266b9;
enum class Result { Disabled, Unrelated, Unsupported, Applied };

struct Builder {
    uint32_t next, type, glsl;
    std::vector<uint32_t> words;

    uint32_t Id() { return next++; }
    void Emit(uint32_t op, std::initializer_list<uint32_t> operands) {
        words.push_back((uint32_t(operands.size() + 1) << 16) | op);
        words.insert(words.end(), operands);
    }
    uint32_t Op(uint32_t op, uint32_t a, uint32_t b) {
        const auto id = Id();
        Emit(op, {type, id, a, b});
        return id;
    }
    uint32_t Ext(uint32_t op, std::initializer_list<uint32_t> args) {
        const auto id = Id();
        words.push_back((uint32_t(args.size() + 5) << 16) | 12);
        words.insert(words.end(), {type, id, glsl, op});
        words.insert(words.end(), args);
        return id;
    }
    uint32_t Dot(const std::array<uint32_t, 3>& a, const std::array<uint32_t, 3>& b) {
        const auto x = Op(133, a[0], b[0]), y = Op(133, a[1], b[1]), z = Op(133, a[2], b[2]);
        return Op(129, x, Op(129, y, z));
    }
};

struct DotTerm {
    uint32_t a, b;
};
using DotTerms = std::array<DotTerm, 3>;

inline bool ReadDot(const SpirvIndex::Index& index, uint32_t id, DotTerms& terms) {
    size_t count = 0;
    const auto flatten = [&](auto&& self, uint32_t value, uint32_t depth) -> bool {
        if (depth > 3) return false;
        if (const auto* add = index.Binary(value, 129))
            return self(self, add[0], depth + 1) && self(self, add[1], depth + 1);
        if (count >= terms.size()) return false;
        const auto* mul = index.Binary(value, 133);
        if (!mul) return false;
        terms[count++] = {mul[0], mul[1]};
        return true;
    };
    return flatten(flatten, id, 0) && count == terms.size();
}

inline std::array<uint64_t, 3> DotKeys(const DotTerms& terms) {
    std::array<uint64_t, 3> keys;
    for (size_t i = 0; i < terms.size(); ++i) {
        keys[i] = uint64_t(std::min(terms[i].a, terms[i].b)) << 32 |
                  std::max(terms[i].a, terms[i].b);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

inline bool SameDot(const SpirvIndex::Index& index, uint32_t id,
                    const std::array<uint32_t, 3>& a, const std::array<uint32_t, 3>& b) {
    DotTerms actual;
    if (!ReadDot(index, id, actual)) return false;
    DotTerms expected;
    for (size_t i = 0; i < 3; ++i) expected[i] = {a[i], b[i]};
    return DotKeys(actual) == DotKeys(expected);
}

struct Site {
    uint32_t result = 0, exponent = 0, half_dot = 0, normalization = 0, gate = 0;
    std::array<uint32_t, 3> normal{}, view{}, light{}, half{};
};

inline bool ReadHalfComponent(const SpirvIndex::Index& index, uint32_t half, uint32_t& view,
                              uint32_t& light, uint32_t& inverse) {
    if (const auto it = index.defs.find(half); it != index.defs.end() && it->second.op == 169) {
        const uint32_t* select = index.words + it->second.at;
        if ((select[0] >> 16) != 6 || select[1] != index.f32) return false;
        if (index.IsConst(select[4], 0))
            half = select[5];
        else if (index.IsConst(select[5], 0))
            half = select[4];
        else
            return false;
    }
    const auto* mul = index.Binary(half, 133);
    if (!mul) return false;
    for (uint32_t side = 0; side < 2; ++side) {
        const auto* sum = index.Binary(mul[side], 129);
        if (!sum || !index.Ext(mul[1 - side], 32, 1)) continue;
        for (uint32_t view_side = 0; view_side < 2; ++view_side) {
            if (!index.Binary(sum[view_side], 133)) continue;
            view = sum[view_side];
            light = sum[1 - view_side];
            inverse = mul[1 - side];
            return true;
        }
    }
    return false;
}

inline bool ReadVectors(const SpirvIndex::Index& index, const DotTerms& terms, Site& site) {
    uint32_t inverse = 0;
    for (size_t i = 0; i < 3; ++i) {
        uint32_t component_inverse = 0;
        if (ReadHalfComponent(index, terms[i].b, site.view[i], site.light[i], component_inverse)) {
            site.normal[i] = terms[i].a;
            site.half[i] = terms[i].b;
        } else if (ReadHalfComponent(index, terms[i].a, site.view[i], site.light[i], component_inverse)) {
            site.normal[i] = terms[i].b;
            site.half[i] = terms[i].a;
        } else {
            return false;
        }
        if (inverse && inverse != component_inverse) return false;
        inverse = component_inverse;
    }
    return true;
}

inline bool FindNormalization(const SpirvIndex::Index& index, Site& site) {
    for (const auto& [id, def] : index.defs) {
        if (def.op != 129) continue;
        const uint32_t* add = index.words + def.at;
        if ((add[0] >> 16) != 5 || add[1] != index.f32) continue;
        for (uint32_t side = 0; side < 2; ++side) {
            if (!index.IsConst(add[3 + side], 0.25f)) continue;
            const auto* mul = index.Binary(add[4 - side], 133);
            if (!mul) continue;
            for (uint32_t mul_side = 0; mul_side < 2; ++mul_side) {
                if (mul[mul_side] == site.exponent && index.IsConst(mul[1 - mul_side], 0.125f)) {
                    if (site.normalization) return false;
                    site.normalization = id;
                }
            }
        }
    }
    return site.normalization != 0;
}

inline bool FindPrimaryGate(const SpirvIndex::Index& index, Site& site) {
    for (const auto& [id, def] : index.defs) {
        const auto* clamp = index.Ext(id, 43, 3);
        if (!clamp || !index.IsConst(clamp[1], 0) || !index.IsConst(clamp[2], 1)) continue;
        const auto* mul = index.Binary(clamp[0], 133);
        if (!mul) continue;
        for (uint32_t side = 0; side < 2; ++side) {
            if (!index.IsConst(mul[side], 16) ||
                !SameDot(index, mul[1 - side], site.normal, site.light))
                continue;
            if (site.gate) return false;
            site.gate = id;
        }
    }
    return site.gate != 0;
}

// Discover the four Blinn lobes and their vectors from def-use structure, independent of IDs,
// module size, checksum and compiler optimization level.
inline Result Patch(uint64_t hash, bool fragment, bool enabled, std::vector<uint32_t>& spv,
                    bool strict = false) {
    if (!enabled) return Result::Disabled;
    if (!fragment || hash != DirectionalHash) return Result::Unrelated;
    const auto index = SpirvIndex::Build(spv);
    if (!index.valid || index.floats != 1 || !index.f32 || !index.glsl || !index.first_function)
        return Result::Unsupported;

    std::vector<Site> sites;
    for (const auto& [result, def] : index.defs) {
        const auto* exp = index.Ext(result, 29, 1);
        if (!exp) continue;
        const auto* mul = index.Binary(exp[0], 133);
        if (!mul) continue;
        for (uint32_t side = 0; side < 2; ++side) {
            const auto* log = index.Ext(mul[side], 30, 1);
            if (!log) continue;
            const auto* clamp = index.Ext(log[0], 43, 3);
            const auto* exponent = index.Ext(mul[1 - side], 40, 2);
            if (!clamp || !exponent || !index.IsConst(clamp[1], 0) || !index.IsConst(clamp[2], 1) ||
                !(index.IsConst(exponent[0], 1) || index.IsConst(exponent[1], 1)))
                continue;
            DotTerms terms;
            Site site;
            site.result = result;
            site.exponent = mul[1 - side];
            site.half_dot = clamp[0];
            if (!ReadDot(index, clamp[0], terms)) continue;
            if (!ReadVectors(index, terms, site)) continue;
            sites.push_back(site);
        }
    }
    if (sites.size() != 4) return Result::Unsupported;
    std::sort(sites.begin(), sites.end(), [&](const Site& a, const Site& b) {
        return index.defs.at(a.result).at < index.defs.at(b.result).at;
    });
    const auto same_view = [&](const Site& a, const Site& b) {
        for (size_t component = 0; component < 3; ++component) {
            const auto* x = index.Binary(a.view[component], 133);
            const auto* y = index.Binary(b.view[component], 133);
            if (!x || !y || std::min(x[0], x[1]) != std::min(y[0], y[1]) ||
                std::max(x[0], x[1]) != std::max(y[0], y[1]))
                return false;
        }
        return true;
    };
    for (size_t i = 0; i < sites.size(); ++i) {
        if (sites[i].normal != sites[0].normal || !same_view(sites[i], sites[0]) ||
            !FindNormalization(index, sites[i]) || (i < 2 && !FindPrimaryGate(index, sites[i])))
            return Result::Unsupported;
    }

    uint32_t boolean = 0, zero = 0, one = 0, two = 0, half = 0, sixteen = 0;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16, op = spv[i] & 0xffff;
        if (op == 20 && size == 2) boolean = spv[i + 1];
        if (op == 43 && size == 4 && spv[i + 1] == index.f32) {
            switch (spv[i + 3]) {
            case 0: zero = spv[i + 2]; break;
            case 0x3f000000: half = spv[i + 2]; break;
            case 0x3f800000: one = spv[i + 2]; break;
            case 0x40000000: two = spv[i + 2]; break;
            case 0x41800000: sixteen = spv[i + 2]; break;
            }
        }
        i += size;
    }
    if (!boolean || !zero || !one || !two || !half || !sixteen) return Result::Unsupported;

    Builder b{spv[3], index.f32, index.glsl, {}};
    const auto function_type = b.Id(), function = b.Id();
    b.Emit(33, {function_type, index.f32, index.f32, index.f32, index.f32, index.f32, index.f32,
                index.f32});
    const auto constant = [&](float value) {
        const auto id = b.Id();
        b.Emit(43, {index.f32, id, std::bit_cast<uint32_t>(value)});
        return id;
    };
    const auto minus_two = constant(-2), sqrt_two = constant(1.4142135623730951f);
    const auto maximum_n = constant(8192), minimum_den = constant(1e-20f);
    auto declarations = std::move(b.words);
    b.words.clear();
    b.Emit(54, {index.f32, function, 0, function_type});
    const auto parameter = [&]() {
        const auto id = b.Id();
        b.Emit(55, {index.f32, id});
        return id;
    };
    const auto exponent = parameter(), view = parameter(), light = parameter(), hdot = parameter();
    const auto view_half = parameter(), primary = parameter();
    const auto entry = b.Id(), dark = b.Id(), visible = b.Id(), merge = b.Id();
    b.Emit(248, {entry});
    const auto greater = [&](uint32_t value) {
        const auto id = b.Id();
        b.Emit(186, {boolean, id, value, zero});
        return id;
    };
    const auto front_v = greater(view), front_l = greater(light), front_h = greater(hdot);
    const auto front_vl = b.Id(), front = b.Id();
    b.Emit(167, {boolean, front_vl, front_v, front_l});
    b.Emit(167, {boolean, front, front_vl, front_h});
    b.Emit(247, {merge, 0});
    b.Emit(250, {front, visible, dark});
    b.Emit(248, {dark});
    b.Emit(249, {merge});
    b.Emit(248, {visible});
    const auto n = b.Ext(43, {exponent, one, maximum_n});
    const auto v = b.Ext(37, {view, one}), l = b.Ext(37, {light, one}), h = b.Ext(37, {hdot, one});
    const auto t = b.Op(136, minus_two, n), c2 = b.Ext(29, {t});
    const auto numerator = b.Op(131, one, c2), map_den = b.Op(131, sqrt_two, c2);
    const auto a2 = b.Op(136, numerator, map_den);
    const auto h2 = b.Op(133, h, h), one_h2 = b.Op(131, one, h2), ah2 = b.Op(133, a2, h2);
    const auto dden = b.Op(129, one_h2, ah2), dden2 = b.Op(133, dden, dden);
    const auto d = b.Op(136, a2, dden2), one_a2 = b.Op(131, one, a2);
    const auto v2 = b.Op(133, v, v), l2 = b.Op(133, l, l);
    const auto vpart = b.Op(133, one_a2, v2), lpart = b.Op(133, one_a2, l2);
    const auto vsum = b.Op(129, vpart, a2), lsum = b.Op(129, lpart, a2);
    const auto vroot = b.Ext(31, {vsum}), lroot = b.Ext(31, {lsum});
    const auto lambda_v = b.Op(133, l, vroot), lambda_l = b.Op(133, v, lroot);
    const auto vis_den = b.Op(129, lambda_v, lambda_l);
    const auto vis_safe = b.Ext(40, {vis_den, minimum_den}), vis = b.Op(136, half, vis_safe);
    const auto brdf = b.Op(133, d, vis), raw_lit = b.Op(133, brdf, l);
    uint32_t lit = raw_lit;
    if (!strict) {
        const auto c = b.Ext(43, {view_half, zero, one}), c_squared = b.Op(133, c, c);
        const auto reference_part = b.Op(133, one_a2, c_squared);
        const auto reference_sum = b.Op(129, reference_part, a2);
        const auto reference_root = b.Ext(31, {reference_sum});
        const auto reference_n = b.Op(129, n, two), reference_width = b.Op(133, reference_n, a2);
        const auto reference_product = b.Op(133, reference_width, reference_root);
        const auto reference_scale = b.Op(133, reference_product, half);
        const auto gate_wide = b.Op(133, c, sixteen), gate_primary = b.Ext(37, {gate_wide, one});
        const auto is_primary = greater(primary), gate = b.Id();
        b.Emit(169, {index.f32, gate, is_primary, gate_primary, one});
        const auto gain = b.Op(133, reference_scale, gate);
        lit = b.Op(133, raw_lit, gain);
    }
    b.Emit(249, {merge});
    b.Emit(248, {merge});
    const auto result = b.Id();
    b.Emit(245, {index.f32, result, zero, dark, lit, visible});
    b.Emit(254, {result});
    b.Emit(56, {});
    auto helper = std::move(b.words);
    b.words.clear();

    struct Target {
        size_t site;
        bool bypass;
    };
    std::unordered_map<uint32_t, Target> targets;
    for (size_t i = 0; i < sites.size(); ++i) {
        if (!targets.emplace(sites[i].result, Target{i, false}).second ||
            !targets.emplace(sites[i].normalization, Target{i, true}).second ||
            (i < 2 && !targets.emplace(sites[i].gate, Target{i, true}).second))
            return Result::Unsupported;
    }
    size_t seen = 0;
    uint32_t view_dot = 0;
    std::vector<uint32_t> patched(spv.begin(), spv.begin() + 5);
    for (size_t i = 5; i < spv.size(); i += spv[i] >> 16) {
        if (i == index.first_function)
            patched.insert(patched.end(), declarations.begin(), declarations.end());
        const auto size = spv[i] >> 16;
        const auto id = size >= 3 ? spv[i + 2] : 0;
        const auto target = targets.find(id);
        if (target == targets.end() || index.defs.at(id).at != i || spv[i + 1] != index.f32) {
            patched.insert(patched.end(), spv.begin() + i, spv.begin() + i + size);
            continue;
        }
        const auto& site = sites[target->second.site];
        if (target->second.bypass) {
            b.Emit(83, {index.f32, id, one});
        } else {
            if (!view_dot) view_dot = b.Dot(site.normal, site.view);
            const auto light_dot = b.Dot(site.normal, site.light);
            const auto vh = b.Dot(site.view, site.half);
            b.Emit(57, {index.f32, id, function, site.exponent, view_dot, light_dot,
                        site.half_dot, vh, target->second.site < 2 ? one : zero});
        }
        ++seen;
        patched.insert(patched.end(), b.words.begin(), b.words.end());
        b.words.clear();
    }
    if (seen != targets.size()) return Result::Unsupported;
    patched.insert(patched.end(), helper.begin(), helper.end());
    patched[3] = b.next;
    spv = std::move(patched);
    return Result::Applied;
}
} // namespace Vulkan::Ggx
