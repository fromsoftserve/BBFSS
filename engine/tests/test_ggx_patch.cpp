// SPDX-License-Identifier: GPL-3.0-or-later
#include "../gpu/shadps4/video_core/renderer_vulkan/vk_ggx_patch.h"
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
    assert(size > 0 && size % 4 == 0);
    std::vector<uint32_t> words(size_t(size) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), size);
    assert(file);
    return words;
}
}

int main(int argc, char** argv) {
    using namespace Vulkan::Ggx;
    assert(argc == 1 || argc >= 4);
    std::vector<uint32_t> malformed{0x07230203, 0x10600, 0, 10, 0, 0};
    const auto original = malformed;
    assert(Patch(DirectionalHash, true, false, malformed) == Result::Disabled);
    assert(Patch(DirectionalHash, false, true, malformed) == Result::Unrelated);
    assert(Patch(0x1234, true, true, malformed) == Result::Unrelated);
    assert(Patch(DirectionalHash, true, true, malformed) == Result::Unsupported);
    assert(malformed == original);
    if (argc >= 4) {
        const std::filesystem::path output = argv[argc - 1];
        assert(!std::filesystem::exists(output));
        std::filesystem::create_directories(output);
        for (int i = 1; i <= argc - 2; ++i) {
            const auto source = Read(argv[i]);
            auto words = source;
            assert(Patch(DirectionalHash, true, false, words) == Result::Disabled && words == source);
            assert(Patch(0x1234, true, true, words) == Result::Unrelated && words == source);
            assert(Patch(DirectionalHash, true, true, words) == Result::Applied);
            auto second = source;
            assert(Patch(DirectionalHash, true, true, second) == Result::Applied && second == words);
            const auto patched = words;
            assert(Patch(DirectionalHash, true, true, words) == Result::Unsupported && words == patched);
            auto strict = source;
            assert(Patch(DirectionalHash, true, true, strict, true) == Result::Applied);
            assert(strict != words);
            auto strict_second = source;
            assert(Patch(DirectionalHash, true, true, strict_second, true) == Result::Applied &&
                   strict_second == strict);
            for (const auto* variant : {&words, &strict}) {
                auto sharpened = *variant;
                assert(Vulkan::GgxLobes::ScaleLightSharpness(
                           sharpened, Vulkan::GgxLobes::SharpnessRemoved) == 4);
            }

            if (i == 1) {
                auto incomplete = source;
                bool removed = false;
                for (size_t at = 5; at < incomplete.size(); at += incomplete[at] >> 16) {
                    if ((incomplete[at] & 0xffff) == 12 && incomplete[at + 2] == 964) {
                        incomplete[at] = (incomplete[at] & 0xffff0000) | 140;
                        removed = true;
                        break;
                    }
                }
                assert(removed);
                const auto bad = incomplete;
                assert(Patch(DirectionalHash, true, true, incomplete) == Result::Unsupported &&
                       incomplete == bad);
            }

            const std::string name = i == 1 ? "legacy" : i == 2 ? "current" : "variant-" + std::to_string(i);
            for (const auto* variant : {&words, &strict}) {
                auto path = output / name;
                path += variant == &strict ? "-strict.spv" : ".spv";
                std::ofstream file(path, std::ios::binary);
                file.write(reinterpret_cast<const char*>(variant->data()),
                           std::streamsize(variant->size() * 4));
                file.close();
                assert(file);
            }
        }
    }
    std::cout << "GGX structural matching, binary gates and deterministic transformations passed\n";
}
