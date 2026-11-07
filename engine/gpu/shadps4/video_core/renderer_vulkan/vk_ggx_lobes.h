// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// Pattern-based specular upgrades for the game's own fragment shaders. Nothing is keyed on
// shader hashes or IDs; each rewrite only fires on the exact gloss-driven expression shapes:
//   Phong local-light lobe: exp2(0.25 * max(1, n) * log2(clamp(R.L, 0, 1)))
//   Blinn lobe:             exp2(max(1, n) * log2(clamp(N.H, 0, 1)))
//   Env reflection fetch:   textureLod(env, dir, min(6, 6.5 - gloss * 13 * 0.5))
// Phong lobes become peak-anchored, width-matched GGX. Specular AA widens lobes by the
// screen-space variance of the half-vector slope (Kaplanyan/Tokuyoshi) while preserving
// energy. Env fetches blend in a blurrier mip to approximate the GGX tail.

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Vulkan::GgxLobes {

struct Options {
    bool local_ggx = true;
    bool specular_aa = true;
    bool env_tails = true;
};

struct Stats {
    uint32_t phong = 0, blinn = 0, env = 0, filtered = 0;
    bool Changed() const { return phong || blinn || env; }
};

// Fraction of a width-matched GGX lobe's energy lying beyond twice its half-energy radius
// that a Gaussian (Blinn) lobe lacks: 1/(1+4) - exp(-4 ln 2) ~= 0.14. Two mips = 4x width.
constexpr float EnvTailWeight = 0.14f;
constexpr float EnvTailMips = 2.0f;

inline Stats Patch(std::vector<uint32_t>& spv, const Options& options) {
    Stats stats{};
    if (spv.size() < 5 || spv[0] != 0x07230203) return stats;
    if (!options.local_ggx && !options.specular_aa && !options.env_tails) return stats;

    struct Def { size_t at; uint32_t op; };
    std::unordered_map<uint32_t, Def> defs;
    std::unordered_map<uint32_t, uint32_t> constants, vectors;
    uint32_t glsl = 0, f32 = 0, entry = 0, floats = 0;
    size_t first_function = 0;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16, op = spv[i] & 0xffff;
        if (!size || size > spv.size() - i) return stats;
        if (op == 11 && size >= 3) {
            const char* name = reinterpret_cast<const char*>(&spv[i + 2]);
            if (std::string_view(name, strnlen(name, (size - 2) * 4)) == "GLSL.std.450")
                glsl = spv[i + 1];
        }
        if (op == 22 && size == 3 && spv[i + 2] == 32) f32 = spv[i + 1], ++floats;
        if (op == 23 && size == 4 && spv[i + 3] == 4) vectors[spv[i + 1]] = spv[i + 2];
        if (op == 15 && size >= 3 && spv[i + 1] == 4 && !entry) entry = spv[i + 2];
        if (op == 43 && size == 4) constants[spv[i + 2]] = spv[i + 3];
        if ((op == 12 || op == 43 || (op >= 129 && op <= 136)) && size >= 3)
            defs[spv[i + 2]] = {i, op};
        if (op == 54 && !first_function) first_function = i;
        i += size;
    }
    if (!glsl || !f32 || floats != 1 || !entry || !first_function) return stats;

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
    // max(1, X) -> result id of the max, or 0.
    const auto gloss_max = [&](uint32_t id) -> uint32_t {
        const auto* a = ext(id, 40, 2);
        if (!a) return 0;
        return (is_const(a[0], 1) && !is_const(a[1], 1)) || (is_const(a[1], 1) && !is_const(a[0], 1))
                   ? id : 0;
    };

    struct Lobe { uint32_t n, c; bool phong; };
    std::unordered_map<uint32_t, Lobe> lobes;
    std::unordered_map<size_t, uint32_t> envs; // instruction offset -> lod id
    for (const auto& [id, def] : defs) {
        if (def.op != 12) continue;
        const auto* e = ext(id, 29, 1);
        if (!e) continue;
        const auto* mul = binary(e[0], 133);
        if (!mul) continue;
        for (int side = 0; side < 2; ++side) {
            const auto* log = ext(mul[side], 30, 1);
            if (!log) continue;
            const auto* clamp = ext(log[0], 43, 3);
            if (!clamp || !is_const(clamp[1], 0) || !is_const(clamp[2], 1)) continue;
            const uint32_t other = mul[1 - side];
            if (const auto n = gloss_max(other)) {
                lobes[id] = {n, log[0], false};
            } else if (const auto* scaled = binary(other, 133)) {
                for (int k = 0; k < 2; ++k) {
                    const auto n = gloss_max(scaled[k]);
                    if (n && is_const(scaled[1 - k], 0.25f)) lobes[id] = {n, log[0], true};
                }
            }
        }
    }
    if (options.env_tails) {
        for (size_t i = first_function; i < spv.size(); i += spv[i] >> 16) {
            const uint32_t size = spv[i] >> 16;
            if ((spv[i] & 0xffff) != 88 || size != 7 || spv[i + 5] != 2) continue;
            const auto vec = vectors.find(spv[i + 1]);
            if (vec == vectors.end() || vec->second != f32) continue;
            const auto* m = ext(spv[i + 6], 37, 2);
            if (!m) continue;
            for (int side = 0; side < 2; ++side) {
                if (!is_const(m[side], 6)) continue;
                const auto* sub = binary(m[1 - side], 131);
                if (!sub || !is_const(sub[0], 6.5f)) continue;
                const auto* half = binary(sub[1], 133);
                if (!half) continue;
                for (int k = 0; k < 2; ++k) {
                    const auto* gloss = binary(half[k], 133);
                    if (is_const(half[1 - k], 0.5f) && gloss &&
                        (is_const(gloss[0], 13) || is_const(gloss[1], 13)))
                        envs[i] = spv[i + 6];
                }
            }
        }
    }
    bool want = !envs.empty();
    for (const auto& [id, lobe] : lobes)
        want |= lobe.phong ? (options.local_ggx || options.specular_aa) : options.specular_aa;
    if (!want) return stats;

    uint32_t next = spv[3];
    std::vector<uint32_t> decl, body, helpers;
    const auto emit = [](std::vector<uint32_t>& out, uint32_t op, std::initializer_list<uint32_t> w) {
        out.push_back((uint32_t(w.size() + 1) << 16) | op);
        out.insert(out.end(), w);
    };
    std::unordered_map<uint32_t, uint32_t> made, splats;
    const auto k = [&](float value) {
        const auto bits = std::bit_cast<uint32_t>(value);
        if (const auto it = made.find(bits); it != made.end()) return it->second;
        const auto id = next++;
        emit(decl, 43, {f32, id, bits});
        return made[bits] = id;
    };
    const auto zero = k(0), one = k(1), half = k(0.5f);

    std::vector<uint32_t>* out = &helpers;
    const auto op2 = [&](uint32_t op, uint32_t a, uint32_t b) {
        const auto id = next++;
        emit(*out, op, {f32, id, a, b});
        return id;
    };
    const auto call = [&](uint32_t inst, std::initializer_list<uint32_t> args) {
        const auto id = next++;
        out->push_back((uint32_t(args.size() + 5) << 16) | 12);
        out->insert(out->end(), {f32, id, glsl, inst});
        out->insert(out->end(), args);
        return id;
    };
    const auto add = [&](uint32_t a, uint32_t b) { return op2(129, a, b); };
    const auto sub = [&](uint32_t a, uint32_t b) { return op2(131, a, b); };
    const auto mul = [&](uint32_t a, uint32_t b) { return op2(133, a, b); };
    const auto div = [&](uint32_t a, uint32_t b) { return op2(136, a, b); };
    const auto fmin = [&](uint32_t a, uint32_t b) { return call(37, {a, b}); };
    const auto fmax = [&](uint32_t a, uint32_t b) { return call(40, {a, b}); };

    const auto fn_type = next++;
    emit(decl, 33, {fn_type, f32, f32, f32, f32});
    // Widen alpha^2 by the filtered slope variance (kappa = 0.18).
    const auto widen = [&](uint32_t a2, uint32_t s2) {
        return fmin(add(a2, fmin(mul(half, s2), k(0.18f))), one);
    };
    // Energy-preserving Blinn/Phong exponent filtering: n' = 2/a2' - 2, scaled by (n'+2)/(n+2).
    const auto filtered_power = [&](uint32_t n, uint32_t c, uint32_t s2, float scale) {
        const auto n2 = add(n, k(2));
        const auto a2 = div(k(2), n2);
        const auto nf = fmax(sub(div(k(2), widen(a2, s2)), k(2)), one);
        const auto e = scale == 1 ? nf : mul(k(scale), nf);
        const auto lobe = call(29, {mul(e, call(30, {c}))});
        return mul(lobe, div(add(nf, k(2)), n2));
    };
    const auto helper = [&](bool phong) {
        const auto fn = next++;
        emit(helpers, 54, {f32, fn, 0, fn_type});
        const auto n = next++, c = next++, s2 = next++;
        emit(helpers, 55, {f32, n});
        emit(helpers, 55, {f32, c});
        emit(helpers, 55, {f32, s2});
        emit(helpers, 248, {next++});
        uint32_t result;
        if (phong && options.local_ggx) {
            // Width-matched alpha^2 for a Blinn exponent n; N.H^2 from R.L by the half-angle identity.
            const auto c2 = call(29, {div(k(-2), n)});
            const auto a2 = div(sub(one, c2), sub(k(1.41421356f), c2));
            const auto a2aa = widen(a2, s2);
            const auto nh2 = add(half, mul(half, c));
            const auto den = add(mul(nh2, sub(a2aa, one)), one);
            result = div(mul(a2, a2aa), mul(den, den));
        } else {
            result = filtered_power(n, c, s2, phong ? 0.25f : 1.0f);
        }
        emit(helpers, 254, {result});
        emit(helpers, 56, {});
        return fn;
    };
    uint32_t phong_fn = 0, blinn_fn = 0;

    // Inline screen-space variance of tan(theta_h); only valid in uniform control flow.
    const auto variance = [&](uint32_t c, bool phong) {
        const auto nh2 = phong ? add(half, mul(half, c)) : fmax(mul(c, c), k(1e-4f));
        const auto slope = call(31, {div(sub(one, nh2), fmax(nh2, k(1e-4f)))});
        const auto dx = next++, dy = next++;
        emit(*out, 207, {f32, dx, slope});
        emit(*out, 208, {f32, dy, slope});
        return add(mul(dx, dx), mul(dy, dy));
    };

    std::vector<uint32_t> patched(spv.begin(), spv.begin() + 5);
    std::vector<uint32_t> merges;
    uint32_t function = 0;
    size_t decl_at = 0;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16, op = spv[i] & 0xffff;
        if (i == first_function) out = &body, decl_at = patched.size();
        if (op == 54) function = spv[i + 2], merges.clear();
        if (op == 248)
            while (!merges.empty() && merges.back() == spv[i + 1]) merges.pop_back();
        if (op == 247 || op == 246) merges.push_back(spv[i + 1]);
        body.clear();
        bool replaced = false;
        const auto lobe = op == 12 && size >= 3 ? lobes.find(spv[i + 2]) : lobes.end();
        if (lobe != lobes.end()) {
            const auto& l = lobe->second;
            const bool ggx = l.phong && options.local_ggx;
            const bool uniform = function == entry && merges.empty();
            const bool aa = options.specular_aa && uniform;
            if (ggx || aa) {
                uint32_t& fn = l.phong ? phong_fn : blinn_fn;
                if (!fn) {
                    out = &helpers;
                    fn = helper(l.phong);
                    out = &body;
                }
                const auto s2 = aa ? variance(l.c, l.phong) : zero;
                emit(body, 57, {f32, spv[i + 2], fn, l.n, l.c, s2});
                (l.phong ? stats.phong : stats.blinn)++;
                stats.filtered += aa;
                replaced = true;
            }
        } else if (const auto env = envs.find(i); env != envs.end()) {
            const auto type = spv[i + 1], result = spv[i + 2];
            const auto core = next++, tail = next++;
            const auto lod = fmin(add(env->second, k(EnvTailMips)), k(6));
            emit(body, 88, {type, core, spv[i + 3], spv[i + 4], 2, env->second});
            emit(body, 88, {type, tail, spv[i + 3], spv[i + 4], 2, lod});
            uint32_t& weight = splats[type];
            if (!weight) {
                weight = next++;
                const auto w = k(EnvTailWeight);
                decl.push_back((uint32_t(3 + 4) << 16) | 44);
                decl.insert(decl.end(), {type, weight, w, w, w, w});
            }
            body.push_back((uint32_t(8) << 16) | 12);
            body.insert(body.end(), {type, result, glsl, 46, core, tail, weight});
            stats.env++;
            replaced = true;
        }
        if (replaced) patched.insert(patched.end(), body.begin(), body.end());
        else patched.insert(patched.end(), spv.begin() + i, spv.begin() + i + size);
        i += size;
    }
    if (!stats.Changed()) return stats;
    std::vector<uint32_t> final_words(patched.begin(), patched.begin() + decl_at);
    final_words.insert(final_words.end(), decl.begin(), decl.end());
    final_words.insert(final_words.end(), patched.begin() + decl_at, patched.end());
    final_words.insert(final_words.end(), helpers.begin(), helpers.end());
    final_words[3] = next;
    spv = std::move(final_words);
    return stats;
}

// The game limits each light's gloss exponent ("specular sharpness"):
//   n = max(1, w * 2^(13g) / (2^(13g) * 0.0100021362 + w)),  i.e. 1/n = 1/2^(13g) + 0.01/w
// with w a per-light sharpness. Scaling w by k equals dividing the 0.0100021362 constant by k,
// so this rewrites only that operand of `exp2(13 * g) * 0.0100021362`. At SharpnessRemoved the
// cap is effectively gone (n -> 2^(13g)); a tiny nonzero constant keeps w = 0 lights at 0/x = 0.
constexpr float SharpnessCap = 0.01000213623046875f;
constexpr float SharpnessRemoved = 100.0f;

inline uint32_t ScaleLightSharpness(std::vector<uint32_t>& spv, float scale) {
    if (!(scale > 1.0f) || spv.size() < 5 || spv[0] != 0x07230203) return 0;
    std::unordered_map<uint32_t, size_t> defs;
    std::unordered_map<uint32_t, uint32_t> constants;
    uint32_t glsl = 0, f32 = 0;
    size_t first_function = 0;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t size = spv[i] >> 16, op = spv[i] & 0xffff;
        if (!size || size > spv.size() - i) return 0;
        if (op == 11 && size >= 3) {
            const char* name = reinterpret_cast<const char*>(&spv[i + 2]);
            if (std::string_view(name, strnlen(name, (size - 2) * 4)) == "GLSL.std.450")
                glsl = spv[i + 1];
        }
        if (op == 22 && size == 3 && spv[i + 2] == 32) f32 = f32 ? ~0u : spv[i + 1];
        if (op == 43 && size == 4) constants[spv[i + 2]] = spv[i + 3];
        if ((op == 12 || op == 133) && size >= 3) defs[spv[i + 2]] = i;
        if (op == 54 && !first_function) first_function = i;
        i += size;
    }
    if (!glsl || !f32 || f32 == ~0u || !first_function) return 0;
    const auto is_const = [&](uint32_t id, float value) {
        const auto it = constants.find(id);
        return it != constants.end() && it->second == std::bit_cast<uint32_t>(value);
    };
    // exp2(13 * g) or exp2(g * 13)
    const auto gloss_base = [&](uint32_t id) {
        const auto it = defs.find(id);
        if (it == defs.end()) return false;
        const uint32_t* e = &spv[it->second];
        if ((e[0] & 0xffff) != 12 || (e[0] >> 16) != 6 || e[1] != f32 || e[3] != glsl || e[4] != 29)
            return false;
        const auto m = defs.find(e[5]);
        if (m == defs.end()) return false;
        const uint32_t* w = &spv[m->second];
        return (w[0] & 0xffff) == 133 && (w[0] >> 16) == 5 && w[1] == f32 &&
               (is_const(w[3], 13) || is_const(w[4], 13));
    };
    std::vector<size_t> sites;
    for (const auto& [id, at] : defs) {
        const uint32_t* w = &spv[at];
        if ((w[0] & 0xffff) != 133 || (w[0] >> 16) != 5 || w[1] != f32) continue;
        for (int side = 0; side < 2; ++side)
            if (is_const(w[3 + side], SharpnessCap) && gloss_base(w[4 - side]))
                sites.push_back(at + 3 + side);
    }
    if (sites.empty()) return 0;
    const float cap = scale >= SharpnessRemoved ? SharpnessCap * 1e-5f : SharpnessCap / scale;
    const uint32_t id = spv[3]++;
    const uint32_t constant[] = {(4u << 16) | 43, f32, id, std::bit_cast<uint32_t>(cap)};
    spv.insert(spv.begin() + first_function, std::begin(constant), std::end(constant));
    for (const size_t at : sites) spv[at + 4] = id;
    return uint32_t(sites.size());
}

} // namespace Vulkan::GgxLobes
