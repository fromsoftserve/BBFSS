// SPDX-License-Identifier: GPL-3.0-or-later
#include "../gpu/shadps4/video_core/renderer_vulkan/vk_ggx_lobes.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
std::vector<uint32_t> Read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    assert(file);
    const auto size = file.tellg();
    std::vector<uint32_t> words(size_t(size) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 4));
    return words;
}
}

// Usage: ggx-lobes-test [cache-dir out-dir]
int main(int argc, char** argv) {
    using namespace Vulkan::GgxLobes;
    std::vector<uint32_t> malformed{0x07230203, 0x10600, 0, 10, 0, 0};
    const auto original = malformed;
    assert(!Patch(malformed, {}).Changed() && malformed == original);
    if (argc == 3) {
        const std::filesystem::path out = argv[2];
        std::filesystem::create_directories(out);
        Stats total{};
        uint32_t shaders = 0;
        for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
            if (entry.path().extension() != ".spv") continue;
            const auto source = Read(entry.path());
            auto words = source;
            assert(!Patch(words, {false, false, false}).Changed() && words == source);
            const auto stats = Patch(words, {});
            if (!stats.Changed()) {
                assert(words == source);
                continue;
            }
            ++shaders;
            total.phong += stats.phong, total.blinn += stats.blinn, total.env += stats.env;
            total.filtered += stats.filtered;
            std::ofstream file(out / entry.path().filename(), std::ios::binary);
            file.write(reinterpret_cast<const char*>(words.data()), std::streamsize(words.size() * 4));
            for (const Options options : {Options{true, false, false}, Options{false, true, false},
                                          Options{false, false, true}}) {
                auto variant = source;
                if (Patch(variant, options).Changed()) {
                    std::ofstream v(out / (entry.path().stem().string() + "_v" +
                                           std::to_string(options.local_ggx * 1 +
                                                          options.specular_aa * 2 +
                                                          options.env_tails * 4) + ".spv"),
                                    std::ios::binary);
                    v.write(reinterpret_cast<const char*>(variant.data()),
                            std::streamsize(variant.size() * 4));
                }
            }
        }
        std::cout << shaders << " shaders: " << total.phong << " phong, " << total.blinn
                  << " blinn, " << total.env << " env, " << total.filtered << " filtered\n";
    }
    std::cout << "GGX lobe patch passed\n";
}
