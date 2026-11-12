// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstdint>
#include <cstdlib>

namespace BbEngine::Batch {
enum Mode : std::uint8_t { Off, On };

inline Mode Switch(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1' ? On : Off;
}
}
