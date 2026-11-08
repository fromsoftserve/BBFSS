// SPDX-License-Identifier: GPL-3.0-or-later
// bbport: the GT7 tone mapping operator (Polyphony Digital's sample, MIT licence, via
// github.com/aletschfirn/gt7tonemapper; notice in gt7_tonemap.cpp) and fitted ACES as alternatives
// to the game's tone curve in its final composite pixel shaders (docs/hdr-scene-path.md).
//
// Composites are compiled with all three: per draw, Gt7Push::mode selects uniform branches for GT7
// and ACES, or the game's curve (its own instructions, bit for bit), so the choice changes live.
// The alternatives need the three channels together; pure curve dependencies are moved ahead of
// the first clamp use when needed, and phi nodes merge the selected channels. Skipped when the HDR
// extension (BB_HDR_SCENE_CURVE=1, tone_curve.h) owns the curve.
#pragma once

#include "common/types.h"
#include "shader_recompiler/native/tone_curve.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader {
struct Pools;
namespace IR {
struct Program;
}
} // namespace Shader

namespace Shader::Native {

/// Adds GT7 and ACES to a composite (a fragment program whose curve is in one block). Sets
/// Info::gt7_checked for any composite and Info::uses_tone_curve when applied; returns whether it
/// applied.
bool ApplyGt7ToneMap(IR::Program& program, const RuntimeInfo& runtime_info, Pools& pools);

/// The composites found in captures (docs/hdr-scene-path.md): a cached record of one without
/// Info::gt7_checked predates the pass and is compiled again.
bool IsKnownComposite(u64 hash);

/// Alternative tone mapper parameters, pushed in ToneCurvePush's place when the HDR extension is off.
struct Gt7Push {
    float mode;     ///< 0: game (bit for bit), 1: GT7, 2: ACES
    float exposure; ///< scene scale before GT7/ACES (1: the game's mid-tones)
    float unused[2];
};
static_assert(sizeof(Gt7Push) == sizeof(ToneCurvePush));

/// bb-shader-lab: 0 skips the pass on this thread, -1 leaves it to the normal rules.
inline thread_local int gt7_lab_override = -1;

} // namespace Shader::Native
