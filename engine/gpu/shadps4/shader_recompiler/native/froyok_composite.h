// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include "common/types.h"

namespace Shader::IR {
struct Program;
}

namespace Shader::Native {

inline constexpr u64 FroyokCompositeHash = 0x5c02eaf8;
inline constexpr u32 FroyokPushOffset = 144;
struct FroyokPush {
    float blend;
    float unused[3];
};
static_assert(sizeof(FroyokPush) == 16);

bool ApplyFroyokComposite(IR::Program& program);
bool FroyokCompositeEnabled();

inline thread_local int froyok_lab_override = -1;

} // namespace Shader::Native
