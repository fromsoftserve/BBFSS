# Ambient occlusion quality (`effect_ssao`, `effect_gtao`, `graphics_ao_quality`)

The launchers' Graphics page and the in-game overlay offer one "Ambient occlusion quality"
choice mapped to the ini keys:

| Choice | `effect_ssao` | `effect_gtao` | `graphics_ao_quality` | What runs |
|---|---|---|---|---|
| Off | `0` | unchanged | unchanged | The game's ambient occlusion is patched out. |
| Default SSAO | `1` | `0` | unchanged | The game's own SSAO shader (`0xcbbca5dc`). |
| GTAO | `1` | `1` | `0` (default) | [`gtao.frag`](../gpu/shadps4/video_core/host_shaders/gtao.frag) replaces the SSAO shader, unchanged (same SPIR-V, same draws). |
| HQ GTAO | `1` | `1` | `1` | See below. |

All three keys are frozen at startup (`startup_effects`, `startup_ao_quality`, like the shadow
resolutions and motion blur quality): the patch and the shader replacement are chosen when the
game starts or pipelines are created or loaded from the pipeline cache, so a change applies after
restarting Bloodborne. The AO strength slider still applies live.

## High

1. **Gather** (`GTAO_SPATIAL_HQ=1 GTAO_SLICES=3 GTAO_STEPS=6`, `gtao_hq_frag`): three rotated
   visibility-bitmask directions with six steps on each side, at the game's AO resolution (half
   the internal resolution). It rejects off-screen samples and keeps a minimum step along the
   slice so near samples don't self-occlude. It keeps Regular's thickness, world reach, strength
   mapping and packed-depth output.
2. **Denoise** ([`gtao_denoise.comp`](../gpu/shadps4/video_core/host_shaders/gtao_denoise.comp)):
   a 5x5 depth/normal-aware filter (two passes through a scratch image) that writes back into the
   gather's own target. It keeps each pixel's packed-depth bytes. The game's bilateral upscale and
   SSGI read the filtered result; no second AO layer is composited. It reads the gather's own
   constants for the normal UVs, so it handles reduced scene proxies and padded targets.
3. **Full resolution**: the draw right after the gather is the game's bilateral upscale
   (one R8 target at the internal resolution, sampling the gather and the R32 depth).
   [`vk_ao_spatial.cpp`](../gpu/shadps4/video_core/renderer_vulkan/vk_ao_spatial.cpp)
   recognises it and overwrites its output. The new output comes from a compute variant of the
   same gather (`GTAO_COMPUTE`, `gtao_full_comp`) run at the upscale target's size, then from the
   denoiser (`FULLRES`). The size follows DLSS/FSR's internal resolution (for example
   1280x720 at Performance and 1504x848 at Balanced for a 2560x1440 output).

   If the upscale can't be matched (a different draw follows, the sampled images differ or the
   draw covers only part of the target), the game's own upscale of the denoised half-resolution
   AO stays, and one line is logged. Unsupported gather targets also log once and keep the
   unfiltered HQ gather.

Every pass is per frame with no history: nothing is reprojected, so there is no ghosting.
Regular allocates no HQ resources and dispatches nothing extra.

## Tests

`gtao-test` ([`test_gtao.cpp`](../tests/test_gtao.cpp)) is a synthetic GPU regression: no game
files are needed.

```
cmake --build engine\out\gpu --target gtao-test settings-test
engine\out\gpu\gtao-test.exe <outdir>             # 257x193 scenes
engine\out\gpu\gtao-test.exe <outdir> --native    # 960x540
engine\out\gpu\gtao-test.exe <outdir> --benchmark # plus paired GPU timestamps
```

It renders planes, corners, curved surfaces, thin rails, perforated foliage and a moved surface.
The Regular pass, a single-direction control, raw HQ and filtered HQ are each compared with a
16-direction, 32-step reference. It also checks that open planes stay unoccluded, that AO off
and the packed depth bytes are exact, and that results repeat without history. Further checks
cover noise reduction, depth/normal edges, odd sizes, partial regions and padded/proxy UVs. The
timings cover synthetic passes only, not a frame. The reference and control shaders are only
built into the test.
