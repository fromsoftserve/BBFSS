// SPDX-License-Identifier: GPL-3.0-or-later
#include "shader_recompiler/native/froyok_composite.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <optional>
#include <set>

#include "bbport_settings.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/native/tone_curve.h"

namespace Shader::Native {
namespace {

IR::Inst* As(const IR::Value& value, IR::Opcode opcode) {
    IR::Inst* const inst = value.TryInst();
    return inst && inst->GetOpcode() == opcode ? inst : nullptr;
}

bool Same(const IR::Value& value, const IR::Inst* inst) {
    return !value.IsImmediate() && value.TryInst() == inst;
}

struct CompositeChannel {
    IR::Value scene;
    IR::Value bloom;
    IR::Inst* stock_addition;
    IR::Value stock_scene;
    u32 stock_addend;
};

bool Either(const IR::Inst* inst, IR::Opcode opcode, const IR::Value& a, const IR::Value& b) {
    return inst && inst->GetOpcode() == opcode &&
           ((inst->Arg(0) == a && inst->Arg(1) == b) ||
            (inst->Arg(0) == b && inst->Arg(1) == a));
}

std::optional<CompositeChannel> StockBloom(const ToneCurveMatch& match,
                                           const IR::Inst* bloom_sample) {
    for (const IR::Use& negative_use : match.result->Uses()) {
        IR::Inst* const negative = As(IR::Value{negative_use.user}, IR::Opcode::FPNeg32);
        if (!negative || negative->NumArgs() != 1 || !Same(negative->Arg(0), match.result)) {
            continue;
        }
        for (const IR::Use& multiply_use : negative->Uses()) {
            IR::Inst* const multiply = multiply_use.user;
            if (multiply->GetOpcode() != IR::Opcode::FPMul32) {
                continue;
            }
            const IR::Value& graded = multiply->Arg(0).TryInst() == negative ? multiply->Arg(1)
                                                                            : multiply->Arg(0);
            bool screen_blend = false;
            for (const IR::Use& addition_use : match.result->Uses()) {
                IR::Inst* const sum = As(IR::Value{addition_use.user}, IR::Opcode::FPAdd32);
                if (!sum || !Either(sum, IR::Opcode::FPAdd32, IR::Value{match.result}, graded)) {
                    continue;
                }
                for (const IR::Use& sum_use : sum->Uses()) {
                    screen_blend |= Either(sum_use.user, IR::Opcode::FPAdd32,
                                           IR::Value{multiply}, IR::Value{sum});
                }
            }
            const IR::Inst* const clamp = As(graded, IR::Opcode::FPSaturate32);
            const IR::Inst* const scale = clamp ? As(clamp->Arg(0), IR::Opcode::FPMul32) : nullptr;
            if (!screen_blend || !scale) {
                continue;
            }
            for (u32 factor = 0; factor < 2; ++factor) {
                IR::Inst* const addition = As(scale->Arg(factor), IR::Opcode::FPAdd32);
                if (!addition) {
                    continue;
                }
                for (u32 addend = 0; addend < 2; ++addend) {
                    const IR::Inst* const product =
                        As(addition->Arg(addend), IR::Opcode::FPMul32);
                    if (!product) {
                        continue;
                    }
                    for (u32 bloom_factor = 0; bloom_factor < 2; ++bloom_factor) {
                        IR::Inst* const extract =
                            As(product->Arg(bloom_factor), IR::Opcode::CompositeExtractF32x4);
                        if (extract && Same(extract->Arg(0), bloom_sample) &&
                            extract->Arg(1).IsImmediate() && extract->Arg(1).U32() < 3) {
                            return CompositeChannel{
                                .scene = match.scene->Arg(1), .bloom = IR::Value{extract},
                                .stock_addition = addition, .stock_scene = addition->Arg(1 - addend),
                                .stock_addend = addend};
                        }
                    }
                }
            }
        }
    }
    return std::nullopt;
}

IR::Value Blend(IR::IREmitter& ir, const IR::Value& original, const IR::Value& bloom) {
    const IR::F32 amount = ir.FPClamp(ir.FroyokParam(0), ir.Imm32(BbSettings::FroyokBlendMin),
                                      ir.Imm32(BbSettings::FroyokBlendMax));
    return ir.FPAdd(IR::F32{original},
                    ir.FPMul(amount, ir.FPSub(IR::F32{bloom}, IR::F32{original})));
}

} // namespace

bool FroyokCompositeEnabled() {
    if (froyok_lab_override >= 0) {
        return froyok_lab_override != 0;
    }
    return BbSettings::Get().startup_bloom_quality == BbSettings::BloomQualityFroyok;
}

bool ApplyFroyokComposite(IR::Program& program) {
    if (!FroyokCompositeEnabled() || program.info.pgm_hash != FroyokCompositeHash ||
        program.info.sw_stage != SwStage::Fragment) {
        return false;
    }
    const auto decline = [](const char* reason) {
        std::printf("Froyok bloom: the display composite is unchanged (%s); native bloom remains available\n",
                    reason);
        return false;
    };
    auto matches = FindToneCurves(program);
    if (!IsComposite(matches)) {
        return decline("the game's tone curve was not recognized");
    }
    IR::Block& block = *matches[0].result->GetParent();
    IR::Inst* bloom_sample = nullptr;
    for (IR::Inst& inst : block.Instructions()) {
        if (inst.GetOpcode() != IR::Opcode::ImageSampleExplicitLod || !inst.Arg(0).IsImmediate()) {
            continue;
        }
        if (inst.Arg(0).U32() == 131074) {
            if (bloom_sample) {
                return decline("more than one bloom texture sample");
            }
            bloom_sample = &inst;
        }
    }
    if (!bloom_sample) {
        return decline("the bloom texture sample was not found");
    }

    std::array<std::optional<CompositeChannel>, 3> channels;
    std::array<u32, 3> match_channels{};
    for (size_t index = 0; index < matches.size(); ++index) {
        const ToneCurveMatch& match = matches[index];
        if (match.result->GetParent() != &block) {
            return decline("the tone curve spans shader blocks");
        }
        auto channel = StockBloom(match, bloom_sample);
        if (!channel) {
            return decline("an original bloom addition was not recognized");
        }
        const u32 component = channel->bloom.TryInst()->Arg(1).U32();
        if (channels[component]) {
            return decline("an original bloom channel is used more than once");
        }
        channels[component] = std::move(channel);
        match_channels[index] = component;
    }
    if (std::ranges::any_of(channels, [](const auto& channel) { return !channel; })) {
        return decline("an original bloom channel is missing");
    }

    for (size_t index = 0; index < matches.size(); ++index) {
        ToneCurveMatch& match = matches[index];
        const CompositeChannel& channel = *channels[match_channels[index]];
        IR::IREmitter at_scene{block, IR::Block::InstructionList::s_iterator_to(*match.scene)};
        match.scene->SetArg(1, Blend(at_scene, channel.scene, channel.bloom));
        channel.stock_addition->SetArg(channel.stock_addend, channel.stock_scene);
    }
    program.info.uses_froyok_bloom = true;
    return true;
}

} // namespace Shader::Native
