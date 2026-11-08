// SPDX-License-Identifier: GPL-3.0-or-later
#include "../gpu/shadps4/video_core/renderer_vulkan/vk_local_falloff.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
std::vector<uint32_t> Read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    assert(file);
    const auto size = file.tellg();
    assert(size >= 0 && size % 4 == 0);
    std::vector<uint32_t> words(size_t(size) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
    return words;
}
}

// Usage: local-falloff-test [dumps-dir out-dir]
int main(int argc, char** argv) {
    using namespace Vulkan::LocalFalloff;
    std::vector<uint32_t> malformed{0x07230203, 0x10600, 0, 10, 0, 0};
    const auto original = malformed;
    assert(!Patch(malformed, 0x7cc2c044).Changed() && malformed == original);
    assert(!Patch(malformed, 0x7cc2c044, false).Changed() && malformed == original);
    std::vector<uint32_t> no_match{0x07230203, 0x10600, 0, 2, 0, (3u << 16) | 14, 0, 1};
    const auto untouched = no_match;
    assert(!Patch(no_match, 0x7cc2c044).Changed() && no_match == untouched);
    if (argc == 3) {
        const std::filesystem::path out = argv[2];
        std::filesystem::create_directories(out);
        Stats total{};
        uint32_t shaders = 0, files = 0;
        for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
            if (entry.path().extension() != ".spv") continue;
            ++files;
            const auto source = Read(entry.path());
            const auto stem = entry.path().stem().string();
            const auto hash = uint32_t(std::stoull(stem.substr(stem.find("0x") + 2), nullptr, 16));
            auto words = source;
            assert(!Patch(words, hash, false).Changed() && words == source);
            const auto stats = Patch(words, hash);
            if (!stats.Changed()) {
                assert(words == source);
                continue;
            }
            ++shaders;
            total.standard += stats.standard;
            total.special += stats.special;
            const auto patched = words;
            assert(!Patch(words, hash).Changed() && words == patched);
            std::ofstream file(out / entry.path().filename(), std::ios::binary);
            file.write(reinterpret_cast<const char*>(words.data()),
                       std::streamsize(words.size() * 4));
        }
        std::cout << files << " files, " << shaders << " shaders: " << total.standard
                  << " standard, " << total.special << " special\n";
    }
    std::cout << "Local falloff patch passed\n";
}
