# BBFSS

**BBFSS is an unofficial, modified fork of [Lantern Engine by Awaken (https://github.com/Awakenchan/LanternEngine)](https://github.com/Awakenchan/LanternEngine)**, focused on enhanced lighting, reflections, shadows and depth of field for Bloodborne. It is not an official Lantern Engine release.

The underlying Lantern Engine runs your own copy of Bloodborne on Windows and Linux, building on bbport and shadPS4. The game's own code runs natively on your CPU; its graphics are translated to Vulkan. BBFSS extends that renderer with the visual effects below. These additions have been developed and evaluated on Windows; equivalent Linux behavior has not yet been verified.

**No game files are included.** You need your own legal copy of Bloodborne (CUSA03173) with update 1.09, dumped from your own PS4.

**BBFSS:** [Repository](https://github.com/fromsoftserve/BBFSS) and [issue tracker](https://github.com/fromsoftserve/BBFSS/issues). Report issues with this modified version here, rather than treating them as upstream bugs.

**Upstream help:** Lantern Engine's [Help & FAQ](https://lantern-engine.netlify.app) covers the original project's setup, graphics settings and troubleshooting. It does not necessarily describe BBFSS-specific effects or behavior.

## Visual enhancements

The effects integrate with the existing Vulkan renderer and the game's depth, normal, lighting and post-processing passes. They do not require editing each map's assets.

### SSGI indirect lighting

Screen-space global illumination adds approximate bounced light and color bleeding from visible surfaces. The implementation uses the visibility-bitmask method described by Therrien, Levesque and Gilet: depth samples cover angular sectors, and newly visible sectors contribute incoming light from the scene.

A half-resolution gather uses reconstructed surface geometry, temporal reprojection and filtered history feedback for approximate additional bounce lighting. Depth-aware denoising and bilateral upsampling bring the result back to scene resolution; the receiving surface's base color tints the indirect contribution. This is a screen-space approximation, not full-scene path tracing: hidden and off-screen surfaces cannot supply reliable light.

Implementation: [`ssgi.comp`](engine/gpu/shadps4/video_core/host_shaders/ssgi.comp).

### Stochastic SSR

Screen-space reflections reconstruct view-space positions from depth and march reflected rays through the visible depth buffer. The fallback fragment path traces frame-varying subpixel samples; the full-resolution compute path traces one ray per pixel. Temporal reprojection and accumulation stabilize the results.

Depth- and normal-aware filtering avoids borrowing reflection directions from unrelated surfaces. Intersections require a front-to-back depth crossing, bisection refinement and a final view-space hit validation. A depth-aware composite blends reflections with the existing environment lighting, with relaxed rough-surface attenuation to cover more materials.

On locally planar, depth-continuous receivers, reflection normals favor the reconstructed geometric plane and limit residual ripple tilt to about 1.15 degrees. This keeps reflected silhouettes coherent while retaining subtle water distortion. The classification is geometric, not water-material identification; irregular surfaces and depth boundaries retain the standard normal-filtering path.

The saved **SSR intensity** slider applies a live 0-3x multiplier after stabilization and roughness fading. Zero restores the environment-reflection contribution rather than replacing it with black. SSR still cannot reflect geometry outside the screen or hidden behind visible objects.

Implementation: [`ssr.frag`](engine/gpu/shadps4/video_core/host_shaders/ssr.frag), [`ssr_trace.comp`](engine/gpu/shadps4/video_core/host_shaders/ssr_trace.comp), [`ssr_temporal.comp`](engine/gpu/shadps4/video_core/host_shaders/ssr_temporal.comp) and the [composite generator](engine/tools/re/ssr_composite_patch.py).

### GTAO

The original SSAO shader is replaced by a GTAO-style screen-space occlusion implementation. It combines GTAO's view-space slice setup with visibility-bitmask angular coverage to estimate how much nearby geometry blocks ambient light.

The replacement preserves the original shader's resource interface and depth-packed output, allowing the existing bilateral upscale to remain in place. The result strengthens small-scale grounding and crevice shading without changing direct illumination. As with other screen-space effects, occlusion is limited by the visible depth buffer.

**Ambient occlusion quality** offers **Regular GTAO** (the unchanged, default shader) and
**High (full-res bitmask GTAO, denoised, experimental)**, which traces three visibility-bitmask
directions at the full internal render resolution (following DLSS/FSR scaling) and applies a
depth/normal-aware denoiser. Set `graphics_ao_quality=1` in `lantern.ini`, or use
either launcher's Graphics page or the in-game Game effects menu, then restart
Bloodborne. Ambient occlusion must also be enabled. Cold compilation and cache
warmup use the same startup-frozen selection; the live AO strength slider still
applies independently.

High samples three distinct directions with six steps on each side, then uses a
5x5 depth/normal-aware spatial filter. Its slice-weight normalization leaves an
unobstructed hemisphere open, and its minimum sample footprint preserves the slice
direction instead of introducing near-sample self-occlusion. These corrections are
HQ-only: High can be lighter than Standard on unobstructed surfaces, rather than
simply darkening the image. It rejects off-screen gather samples and retains the
existing thickness, world reach, strength mapping and native bilateral upscale.

The filter preserves each center pixel's packed-depth bytes and updates the same
AO target consumed by the native upscale and SSGI; it does not composite a second
AO layer. It uses the gather's actual constants for normal UVs, including reduced
scene proxies and padded targets. Unsupported targets/views warn once and retain
unfiltered HQ AO. Standard allocates no HQ GPU resources and dispatches no filter.

This is a spatial-only prototype at the game's AO-pass resolution, **not** a
full-resolution or temporally accumulated AO mode. It adds GPU work. Synthetic
GPU comparisons against a 16-direction, 32-step-per-side screen-space reference
cover planes, corners, curved surfaces, thin rails, perforated foliage and a moved
surface. A corrected single-direction control separates the coverage/filter gain
from the brightness corrections. These checks do not establish in-game appearance,
stability during camera motion, or whole-game FPS; those still need a live A/B
evaluation before a release.

Build/run the standalone shader regression with
`cmake --build engine\out\gpu --target gtao-test settings-test`, then
`engine\out\gpu\gtao-test.exe` and `engine\out\gpu\settings-test.exe`.
The GPU test checks reference-error thresholds, open-plane visibility, AO-off
identity, byte-exact packed depth, repeatability without history, noise reduction,
depth/normal edges, odd sizes, partial regions and padded/proxy UV mapping.
`gtao-test.exe engine\out\ao-hq-analysis\gpu --benchmark` additionally writes
synthetic packed outputs and measures paired, warmed GPU intervals around the
Standard pass and HQ gather plus both filter passes. Disable Vulkan validation
layers for timing and check for competing GPU work; uploads/readbacks and pipeline
compilation are outside the timestamp intervals. Reference/control shaders and the
test executable are not part of the runtime package.
Use `--benchmark-native` instead for a 960x540 synthetic AO target. Neither timing
mode measures a complete frame or includes a new in-game comparison.
`gtao-test.exe engine\out\ao-hq-analysis\native --native` runs the same correctness
checks at 960x540 without timestamps, suitable for synchronization validation.

Implementation: [`gtao.frag`](engine/gpu/shadps4/video_core/host_shaders/gtao.frag),
[`gtao_denoise.comp`](engine/gpu/shadps4/video_core/host_shaders/gtao_denoise.comp) and
[`vk_ao_spatial.cpp`](engine/gpu/shadps4/video_core/renderer_vulkan/vk_ao_spatial.cpp).

### Volumetric lighting

Directional volumetrics integrate scattering along camera rays through a three-dimensional fog volume, using the game's shadow information to form lightshafts. Celestial-direction alignment, moderated light-color saturation and depth-aware reconstruction help the fog follow the scene's lighting without excessive color tint or gaps around thin foliage.

Point lights also contribute genuine volumetric scattering: camera rays are intersected with each light's spherical influence and scattering is integrated within that volume, using inverse-square falloff (smoothly windowed to zero at the light's radius) so fog is densest around the source and fades physically outward. These are not screen-space glow sprites. Point-light fog is deliberately unshadowed; scene depth stops the viewing ray, but walls do not block light traveling into the volume, so light leakage remains possible.

Separate saved sliders control directional and point-light volumetric intensity.
Point-light fog has a live on/off toggle in the Picture section, with live Glow size (how tight or wide the inverse-square glow is) and intensity sliders.
Point-light integration respects each native light-grid quad and scissor, so a light
listed in multiple tiles is not added repeatedly across the whole screen.

The native Windows start includes the previously optional Performance Patch behavior
(renderer defaults, profiling bypasses and shorter worker waits). Its light-grid tile
size follows the selected output resolution automatically; old Performance Patch and
resolution-specific Light Grid selections are ignored. This preserves that patch's
native visual changes, not just its scheduling changes. Explicit effect settings,
including SSR, and the directional/local shadow quality controls remain authoritative.

Implementation: [`volumetric_light.comp`](engine/gpu/shadps4/video_core/host_shaders/volumetric_light.comp) and [`point_volumetric.comp`](engine/gpu/shadps4/video_core/host_shaders/point_volumetric.comp).

### NVIDIA PCSS soft shadows

An implementation of NVIDIA's **Percentage-Closer Soft Shadows (PCSS)** technique replaces the original fixed-width directional shadow filtering. It searches the existing cascaded shadow map for blockers, estimates penumbra width from the separation between blockers and the receiving surface, then performs percentage-closer filtering with that width.

Shadows remain tighter near contact and soften as the receiver moves away from the blocker. This implements the published technique in Vulkan shaders; it is not an RTX feature or an NVIDIA-only SDK dependency.

Implementation: the PCSS path in [`contact_shadow.comp`](engine/gpu/shadps4/video_core/host_shaders/contact_shadow.comp).

### Directional shadow filtering and cascade transitions

The native four-cascade layout and each map's authored split distances are retained.
The in-game **Game effects (restart required) > Shadow resolution** dropdown selects
**High** (2048 atlas, 1024 per cascade), **Very high** (4096 atlas, 2048 per cascade; default)
or **Ultra** (8192 atlas, 4096 per cascade). The selection is saved as
`graphics_shadow_resolution=0/1/2` and applies when Bloodborne restarts, independently
of Picture tuning. It overrides map-specific atlas quality without changing cascade
distances. Ultra uses approximately 256 MiB per D32 atlas versus 64 MiB for Very high,
excluding other working resources, and increases shadow rendering cost.
After the native shadow blur chain, an atlas-safe eight-tap PCF resolve replaces
only the final shadow alpha, preserving the packed depth in RGB. Each tap uses
bilinearly interpolated depth comparisons with receiver-plane correction, rather
than blurring shadow boundaries repeatedly in screen space. Its 1.5-atlas-texel
disk radius gives regular PCF edges a modest softness without changing PCSS.

The optional **PCSS** effect uses the same cascade-safe comparisons, a
receiver-plane-corrected blocker average, and a bias capped by the world-space
shadow texel size. All reads stay inside the selected cascade tile.
Its blocker search uses 64 disk samples including the receiver's own shadow
texel, reducing missed thin casters. Filtering selects 16, 32 or 64 bilinear
comparison taps for radii up to 1, up to 4, or above 4 atlas texels respectively.
Quality depends on footprint, not potentially misleading all-lit/all-shadow
sample prefixes. Contact hardening has a half-texel filtering floor rather than
a fixed world-space softness minimum (`BB_PCSS_MIN` can still override it).
The sample pattern remains frame-stable; no temporal reconstruction is added.
This is our Vulkan implementation of classic PCSS with these enhancements,
not NVIDIA's ShadowWorks PCSS+ SDK.
A guarded native camera hook extends cascades 1-3 backwards by 10% of the preceding
interval for both caster culling and projection. The resolve blends over those
bands only when that hook is installed. Map split distances, total shadow distance,
native stabilization settings are not overridden.

Set `BB_DIRECTIONAL_SHADOWS=0` before launch to restore the previous native blur
path and disable added cascade overlap. The existing PCSS and contact-shadow
toggles still apply; disable those as well for the original directional filtering.

### Hunter's Dream sky seam

A default-on GPU UV correction closes the inset horizontal gap in the verified
Hunter's Dream sky dome. It leaves the panorama texture, vertical UVs, clouds and
moon unchanged; it does not address zenith texture banding.

The correction requires the known shader pair, draw counts and vertex-stream
layout, plus a GPU fingerprint of the first 485 position/UV records. Nonmatching
streams retain their original UVs, and devices without dynamic vertex input keep
the original rendering path. Set `BB_SKY_SEAM_FIX=0` before launch to disable it
for comparison. Normal rendering performs no CPU readback or GPU wait.

Implementation: [`sky_seam.comp`](engine/gpu/shadps4/video_core/host_shaders/sky_seam.comp).
The `sky-seam-test` GPU test accepts a local 485-record, 24-byte position/UV fixture;
no game geometry or texture fixtures are included in this repository.

The separate **experimental zenith debanding prototype** is off by default.
Set `BB_SKY_ZENITH_FIX=1` before launch to try it with the seam correction enabled.
It additionally checks the panorama's size, format, 12-mip layout and 16 measured
base-mip color samples. The initial wrap-aware local bilateral filter (base-mip
rows 0-96) did not noticeably improve the radial streaks in the live comparison.
A stronger diagnostic mode reconstructs a zenith gradient from each row's average
linear-light color and smooths it vertically. A partial blend fades continuously
over base-mip rows 0-224 using a curve with zero first and second derivatives at
both ends, without a brightness-dependent boundary. Its default diagnostic
strength is 40%, so at least 60% of the original panorama contribution is retained
even at the pole. This reduces rather than eliminates authored horizontal
variation; it cannot recover lost source detail. Full-strength reconstruction
looked too flat and exposed a transition cutoff; the user confirmed the
reduced-strength balance in a same-camera original/reconstructed comparison.

Each native mip is processed independently into an owned RGBA16F image; alpha
and the lower panorama are copied rather than filtered. The original sky sampler,
moon, separate cloud draws and game files are unchanged. This prototype rebuilds
the filtered mip chain per supported draw; its rendering cost has not been
benchmarked.

For diagnostic launches only, `BB_SKY_ZENITH_MODE_FILE` can name a text file
containing `0` (original), `1` (local filter), `2` (deliberately flat-color texture
isolation) or `3` (gradient reconstruction), optionally followed by gradient
strength from 0 to 1 (for example, `3 0.4`). The mode is polled every 30 supported
draws, allowing stationary comparisons without restarts. Without that file,
`BB_SKY_ZENITH_FIX=1` selects the initial local filter;
`BB_SKY_ZENITH_ISOLATE=1` additionally selects flat-color isolation. Neither the
mode file nor these experimental flags are set by a normal launcher run.

`sky-zenith-test` accepts a local decoded 2048x512 RGBA8 panorama fixture and checks
GPU smoothing, wrap-boundary behavior, lower-region/alpha preservation, mismatch
passthrough, row-average reconstruction, gradual partial blending against a
numerical reference, retained texture variation, flat-color isolation and repeated use.
Fixtures remain local and are not distributed.

For native tessellation research only, `BB_TESS_EFFECT_MODE_FILE` can name a text
file containing one digit: `0` restores all draws; `1`, `2` or `3` temporarily hides
one captured blended effect family, respectively HS/DS `8453cd1c/7f03e945`,
`bc234799/0c1888bd` or `bc234799/52d756df`. Matching also requires the captured
fragment shader, one control point, depth testing without depth writes, and blending.
The file is polled every 30 matching draws. Invalid or unavailable files restore
all draws and log the error. Explicit frame captures record the omitted draws.
This diagnostic is off by default and changes neither tessellation factors nor
POM or ordinary mesh rendering. It is not a graphics-quality or performance option.

`BB_POM_DRAW_MODE_FILE` optionally names a single-line diagnostic file containing
a known POM fragment hash in hexadecimal and an index count in decimal, such as
`053bbf3b 6894`. It temporarily omits matching indexed, non-tessellated,
depth-tested/depth-writing draws; this selection is not a persistent material identity.
`0 0` restores all draws. The file is polled every 30 supported POM draws; invalid
or unavailable files restore draws and log errors. Explicit captures identify omissions.
The normal launcher does not enable this surface-identification diagnostic.

`BB_POM_PATCH_CAPTURE=1` enables an explicit-capture-only diagnostic for the
identified 210-index paving draw (`c829ee58/08ccd4c7`). With `BB_CAPTURE_DIR`
set and a frame capture active, it writes a new `pom-patch-N` directory containing
the bound GPU index, vertex and constant buffers, plus draw/attribute metadata,
constant-buffer byte adjustments and image descriptors. It stalls the GPU for
readback; normal rendering is unchanged and the normal launcher leaves it disabled.
Texture pixels are not included. Shader hashes and index count identify a research
fixture, not a safe production displacement target. Keep these game-derived dumps
local and ignored; they are inputs for offline geometry/material analysis, not a
mesh-tessellation feature.

`BB_POM_PATCH_CAPTURE=2` additionally captures the candidate neighboring draws:
the 4455-index paving draw (`c829ee58/08ccd4c7`) and the 30096-index ordinary
mesh draw (`9da6aacb/ca7226ec`) observed using the same object constant buffer.
It writes separate `pom-neighbor-N` directories with each draw's actual hashes,
index count, GPU buffers, image descriptors and guest sampler descriptors.
Shared constants and draw order identify candidates only: adjacency and raised-stone
identity must be verified from their indexed geometry. Sampler metadata describes
guest state, not host anisotropy/LOD overrides. This mode is also explicit-capture-only
and default-off; it neither displaces geometry nor changes POM or shadow rendering.

`BB_POM_PATCH_CAPTURE=3` captures the same three draws into `pom-material-N`
directories and additionally reads back the patch's three bound BC4/BC1 GPU images,
including all backing mips as tightly packed compressed blocks. Metadata records
each mip's extent/offset, the exact matched image-view range/swizzle, viewport and
effective sampler parameters after host anisotropy and LOD-bias handling.
Unsupported views/formats are explicitly reported as incomplete captures.
This diagnostic stalls for readback and restores sampled image layouts; it does
not activate displacement. The resulting proprietary fixtures must remain local
and ignored.

Offline material comparisons must distinguish exact compressed GPU bytes from
decoded/sampled equivalence: BC channel decoding and texture interpolation can
differ from a software integer decoder. The local Vulkan sampling probe verifies
all uploaded mip bytes by readback, uses GPU texel-fetch controls to isolate decoding,
and compares the captured anisotropic sampler against a separate bilinear mip-zero
sampler. Constant-UV, repeat-wrap and high-gradient controls precede interpretation
of the patch results. Its height-field images are diagnostics, not native shaded
Bloodborne/POM renders: ray-marched offsets, native derivative reconstruction,
full-scene visibility and physical displacement/residual-POM calibration remain
outside that comparison. Do not use it to justify enabling live displacement.

The separate local native-POM probe executes the translated packed-input vertex
shader and the fragment shader through its POM UV-offset result, replacing only
subsequent shading with floating-point diagnostics. Original-game and BBFSS-patched
references must be kept distinct. Validate finite offsets, a zero-amplitude control,
the inactive secondary ray, and native pre-march UV/material-blend agreement before
fitting geometry to those offsets. A local parallel-plane depth fit is not physical
material height, and halving amplitude need not halve every ray-marched offset.
Offline candidate corrections are not installed or activated in the game.

The POM enhancement retains the top surface as the upper hit bracket when the
dithered ray crosses on its first step. Later hits retain their sampled predecessor.
This prevents the synthetic initial height and dithered layer from producing equal
signed distances and a zero refinement denominator. The correction remains gated
to the three exact original shader modules, preserves `BB_POM_ORIGINAL`, and does
not change displacement or shadow rendering. Supported POM fragment caches use
binary version 9 so old enhanced modules rebuild; unrelated shader caches remain
reusable.

`pom-patch-test [original-053bbf3b.spv original-505acc58.spv original-08ccd4c7.spv output-dir]`
checks first/later bracket math without assets. Optional local original modules
add exact checksum-gate, injected bracket, unrelated-input and idempotence checks
for all three variants and write patched modules for external SPIR-V validation.
Game-derived modules and GPU regression fixtures must remain local and ignored.
The existing `motion-shader-test` also covers metadata round trips: POM fragments
require version 9, while unrelated fragments and ordinary/motion vertices retain
their existing cache versions. Set `BB_POM_ORIGINAL=1` when running
`pom-patch-test` with fixtures to verify that all three native modules remain unchanged.

Offline residual-POM budgeting must compare the corrected native target UV against
the perspective-interpolated UV of displaced geometry at the same screen sample.
Pair with a flat replay to separate rasterization error from geometric movement,
and report coverage/parent-triangle changes separately. The current guarded
prototypes consume about 1.7%, 5.8%, and 17.4% of measurable native relief on
average at illustrative amplitudes 0.003, 0.01, and 0.03 model units, but these
are not recommended global POM reductions. Many samples move even where the
native POM offset is exactly zero; retaining or reducing inward-only residual POM
cannot cancel that geometric overshoot. The larger prototype also overshoots
some samples with measurable native relief.

The ignored local `pom_residual_budget.py` records per-sample required residual
UVs, original-ray fractions, transverse mismatch, and repeatable diagnostics.
This is a geometric accounting experiment, not a native shaded/depth/shadow
replacement or a displaced-surface POM replay. Mesh displacement remains disabled.

The subsequent ignored `pom_hit_guard.py` bounds each subdivided triangle by its
minimum visible corrected native-hit depth, applies the strictest adjoining cap
to shared vertices, and pins triangles without captured hit evidence. Existing
boundary/contact/clearance limits remain upper bounds. Additional pinning removes
measured original-parent changes. In this camera, the saved prototypes retain
1394, 1146, and 1047 moved vertices, affecting about 20.7%, 17.5%, and 16.3% of
original patch area at the three amplitudes. Recomputed residual budgets contain
no overshoot below -1e-6 original-ray fraction across 341276 corresponding samples;
geometry UV movement at exactly zero native-offset samples is zero. Small negative
fractions already present in the native reference are not clamped away.
Detected neighbor-intersection pairs remain unchanged and no triangle flips occur.

These native-hit caps are conservative, view-specific evidence, not a
camera-independent displacement solution. They retain only about 0.67%, 1.85%,
and 5.20% of measurable native relief on average, still leave transverse UV
mismatch, and have not rerun native POM on the displaced surface. The next
evaluation must use actual displaced-geometry GPU inputs and the corrected native
ray march; the accounting alone does not establish native shading, depth, shadow,
or multi-view correctness.

The ignored `pom_displaced_native.py` now replays corrected native POM on the
hit-guarded meshes with captured GPU textures, samplers, constants, and camera.
Its native VS fetch is adapted to float raw-valued attributes to avoid integer
UV requantization during subdivision; original vertices reproduce the packed-input
GPU result exactly. Native vertex transform/tangent arithmetic and the corrected
fragment ray march remain in use. Original tangent attributes are interpolated,
not recomputed for the displaced surface. Full/half/zero amplitude runs remain
diagnostics, not a spatial residual-POM implementation.

Flat subdivision itself changes some ray-march decisions: in this capture,
69 common pixels differ by over 0.001 UV in POM offset, despite a maximum
pre-march UV difference below 0.00002. Keep this baseline separate from displacement
effects. The displaced meshes retain the flat GPU coverage (341275 pixels),
produce finite outputs and exact zero-amplitude offsets, and change material
blend and native rays. Required residuals projected onto the displaced ray still
have transverse mismatch. There are 631 tiny negative projected residuals per
prototype, with reverse offsets below 0.0000153 UV; none exceeds the diagnostic
0.00002 UV threshold. This threshold is not a visual acceptance criterion.
Keeping full POM does not preserve the combined flat target; halving or disabling
POM is not justified either. Isolate the flat-subdivision baseline discrepancy
before evaluating a spatial residual shader. No live displacement is enabled.

The subsequent ignored `pom_flat_isolate.py` diagnostics preserve native POM
offsets and pre-march UVs bit-for-bit while recording gradients, first sampled
height/layer, material blend, and refinement distances. Of the 69 flat-subdivision
samples with offset drift above 0.001 UV, 62 change first-crossing status, 14 change
layer count (these categories overlap), and none changes the POM LOD gate or
pixel dither. At the worst sample, height changes from 0.97003126 to 0.96974134
around a layer of 0.96991491, switching from zero offset to a deeper hit.
This is discrete ray-march sensitivity to small interpolated-UV/filter changes,
not a remaining nonfinite-refinement failure.

`pom_flat_freeze.py` isolates three focal samples by freezing original GPU-measured
march inputs. UV-plus-gradient controls remove the worst sample's discrepancy;
gradient-only controls remove nearly all of another sample's discrepancy. Freezing
UVs, both material gradients, blend, rays, and layer count removes all three
differences exactly while preserving each original focal result. These are local
causal diagnostics, not evidence that every pixel or camera is explained, nor a
proposed global input freeze. The native VS uses affine attribute/transform
arithmetic, so adding triangles changes finite-precision raster interpolation and
quad-gradient evaluation rather than introducing a new nonlinear vertex operation.
Do not change production POM thresholds or suppress outliers to disguise this
baseline. A parent-preserving flat-input control is needed before separating
displacement contributions in a spatial residual-POM experiment.

The ignored `pom_parent_control.py` provides a parent-sourced flat baseline using
a screen-space GPU snapshot of 20 native march scalars: UVs, both gradients,
rays, blend, POM LOD, layer count, and previous-height seed. Inputs are captured
from the original triangles, uploaded to a read-only buffer, and fed into the
executing native march on the refined flat mesh. This does not copy original
POM outputs. All MRT results match the original exactly on 341274 shared pixels
with matching CPU-checked original-parent provenance, restoring all 69 large
flat-subdivision discrepancies. Three original pixels lack refined coverage and
one refined pixel lacks an original snapshot; no equality claim includes them.
A zero-replayed-ray control produces exact zero offsets while retaining original
pre-march UVs.

This control is a camera-specific input snapshot, not analytic parent interpolation,
not a deployable geometry solution, and not displaced-surface residual POM.
It establishes a drift-free reference for controlled input changes; displacement
must next introduce its measured UV/ray changes explicitly instead of freezing
away the effect being evaluated. Production POM, geometry, and live settings are
unchanged by this experiment.

The ignored `pom_spatial_residual_probe.py` adds displaced-minus-refined-flat GPU
input deltas to that parent reference, updating UVs, both gradients, rays, blend,
and POM LOD while retaining original layer count and previous-height seed.
Blend is constrained to [0,1], with affected samples explicitly counted. This is
a synthetic, camera-specific experiment, not a replacement for native interpolation.

An oracle diagnostic estimates per-pixel ray-strength reduction from the captured
target UV and the full-POM hit. The corrected native march then reruns with those
scaled rays; offsets are not simply scaled after rendering. At illustrative
amplitudes 0.003, 0.01, and 0.03, average moved-region UV-target error falls from
0.00015334/0.00045557/0.00131399 to
0.00001173/0.00002949/0.00006377. These are UV errors, not screen-pixel or visual
quality scores. Reduction-only constraints and zero-hit cases are reported, not
silently treated as solved. Hundreds of samples worsen, and local errors remain
as large as about 0.0049 UV because rerunning the march changes discrete hits.
The estimate is not a production solution: it depends on the captured original
target, omits layer-count changes, and does not resolve transverse mismatch,
multi-view behavior, native shading, depth, or shadows. Zero-ray controls remain
exactly zero. No live displacement or residual-POM mode is activated.

The ignored `pom_residual_isolate.py` separates remaining oracle errors from a
current-hit estimate that does not consume the original/flat native POM target.
For the middle oracle prototype, 108 of 165 residual errors above 0.001 UV change
the accepted march iteration, while none changes first-crossing status. Some
errors require strength above one or have zero full hits; these categories
overlap. All instrumentation preserves offsets/UVs exactly on the 341274 shared
parent-reference samples. The one newly covered edge pixel has no parent snapshot;
zero-filled inputs give it a zero layer count and nonfinite output in both the
original and instrumented hybrid replay. It is not a new valid-input refinement
failure, and no scored result or finite-output claim includes that invalid pixel.

The current-hit estimate subtracts CPU-rasterized geometric UV movement from
the displaced surface's own full-POM hit, then estimates a reduction-only ray
strength. The native march reruns with that surface's captured rays; UVs, gradients,
blend, layer count, and other fragment inputs execute normally. Full-ray controls
reproduce the existing native output exactly. Unlike the oracle, the estimator
does not read the original target; a separate flat native replay is used only to
score its result. Average moved-region UV-target error improves by about
74%, 75%, and 77% at the three illustrative amplitudes, but 8010, 6504, and 5429
samples respectively worsen. Zero-hit cases still cannot supply a strength ratio.
This requires an extra full ray march and the underlying geometry remains
camera-specific; neither performance nor reference-independent geometry is
established. Ray rescaling changes the sampled height field instead of correctly
accounting for the displaced layer origin.

The ignored `pom_layer_origin.py` completes that comparison with a projected
layer-origin model. It retains the current full rays and native material sampling,
projects CPU geometric UV movement onto the ray to estimate consumed layer depth,
and samples the actual origin height before marching. The new positive-origin
loop refines only a valid above/below bracket; origins at/inside the sampled height
have an explicitly diagnosed zero residual. Zero origin runs the unchanged native
march and reproduces its output exactly. Neither the estimate nor the uploaded
inputs consume the original POM target or current full hit. This remains a
parallel-layer UV approximation, not a verified physical material-depth model.

Constant-height GPU controls reproduce the expected residual depth, including
exact-zero output for origins at/below the synthetic surface. Both restarted and
original-phase-aligned layer grids are evaluated. Across 341274 scored samples,
the aligned model reduces average moved-region error by roughly 74%, 75%, and
77%, but still worsens 8020, 6501, and 5413 samples at the three illustrative
amplitudes. It improves mean error by less than 0.5% relative to ray shrinking,
with maximum errors of approximately 0.00294, 0.00518, and 0.00744 UV. Preserving
the layer phase therefore does not resolve the remaining mismatch. Those UV
thresholds are diagnostics, not established visual acceptance criteria.

**Research stopping point:** do not integrate or enable cobblestone mesh
displacement/residual POM from these prototypes. Retain native geometry and
native POM with the separately verified first-hit bracket repair. That repair
and cache migration are implemented in the worktree; they have not been installed
in the running game. Displacement remains parked: its guards depend on one captured
view, individual samples regress, transverse movement remains, and normals,
depth, shadows, multi-view stability, and performance have not been established.
The local ignored fixtures and experiment reports preserve the evidence for
future work without requiring further incremental experiments now. No live mode,
installation, commit, or publication is part of this research conclusion.

`billboard-probe-test native-hull.spv native-domain.spv [--benchmark]` is an
offline instanced-quad experiment for the captured flat-billboard pair
`bb88db5f/9071184b`. It runs the actual local hull/domain modules with synthetic
post-local-shader records, then compares their rasterized outputs with a four-vertex
instanced strip. Four floating-point targets expose the three fragment inputs and
fragment position. Checks cover indexed ordering, base vertex, first index,
first instance, clipping, negative/zero extents and overlapping/repeated records.
The optional paired GPU timestamp benchmark uses a synthetic four-target workload;
it does not establish game-frame savings. Native shaders stay local and are not
distributed. This prototype is not wired into game rendering; native local-stage
record preparation, actual material shading/blending and live costs still need
verification before a renderer replacement.

### GGX specular investigation

Offline inspection confirms that the captured native direct lighting is not one
uniform Blinn-Phong implementation. Directional shader `d92266b9` uses normalized
half-vector power lobes and Schlick Fresnel, with up to four profile-sourced lights.
The fourteen known one-to-twelve-light shader variants include both half-vector
Blinn-Phong and reflection-vector Phong lobes. Captured spotlight shader `82951fa5`
also contains both forms. These paths accumulate separate diffuse and specular
targets; their diffuse-wrap/backlighting powers are not specular replacement sites.

Existing materials already supply world normals, specular RGB and smoothness.
Native lighting squares the encoded specular RGB and expands smoothness into an
exponential gloss power, then broadens that power using per-light/profile values.
Smoothness is therefore not GGX roughness. The directional-lighting profile tag
and SSS profile tag belong to different G-buffer alpha channels; neither is a
universal asset identifier. No new metalness texture is required to investigate
a specular-color/gloss workflow.

GGX is technically feasible, but preserving highlight width and preserving peak
brightness are different conversions. An isolated normalized-lobe comparison
over 4096 exponents shows that width-matched GGX has approximately 40% lower peaks
for highly glossy lobes. This is an analytical mapping result, not a captured
image-quality or performance result. Native light-width broadening, local lights'
different Fresnel treatment, and artistic radiance scaling also require calibration.
GGX distribution alone is not a full microfacet BRDF: masking-shadowing visibility
and Fresnel must be handled consistently, and indirect reflections need their own
roughness/filtering treatment before claiming a complete GGX/PBR conversion.
See [Filament's specular BRDF reference](https://google.github.io/filament/Filament.md.html)
for the distribution, visibility and Fresnel distinction.

The current experimental scope is a default-native, offline direct-specular prototype
with an exact native replay baseline. Preserve diffuse/SSS, light colors and
falloff, visibility, geometry, POM and exposure; reject unknown shader/layout
variants rather than replacing arbitrary power instructions. Integration
must select the same model for cold compilation and cached pipelines without
contaminating the native fallback. Existing environment/SSR behavior should remain
explicitly native during that first experiment. The ignored `specular_audit.py`
and `specular-analysis` outputs retain local evidence. The opt-in renderer setting
is described below; offline investigation does not modify an installed engine.

`engine\tools\re\ggx_directional_patch.py` copies an exact supported native module
by default. Its explicit `--mode ggx-width` and `--mode ggx-peak` candidates replace
only the four directional specular lobes with GGX distribution, height-correlated
Smith visibility and incident cosine, retaining native Schlick Fresnel and authored
weights. These are different brightness/width calibrations, not a selected game
setting. Unknown binaries, unexpected replacement sites and existing output files
are rejected. Both supported native module layouts and both candidates pass SPIR-V
validation; generator regression checks also cover native byte preservation,
overwrite refusal and the width mapping across the native gloss range.

The default-off `BB_SPECULAR_CAPTURE=1` diagnostic captures the native fullscreen
draw's complete lighting table, G-buffer images, geometry, push data, depth plane
and packed lighting targets before/after drawing. It also saves the exact SPIR-V
submitted at native shader-module creation on cold and warmup paths, the static
pipeline state and the actual post-update dynamic draw state. New pipeline
snapshots are keyed by the complete pipeline key, not just the shader pair.
Inputs, shaders, cloned saves and replay results must remain local in ignored
directories; no game data is embedded in the tools. Set `BB_GPU_USER_DIR` to an
**existing** isolated directory before launching: a nonexistent directory falls
back to the platform's shared shadPS4 user directory. The local capture helper
now creates that directory before launch; earlier captures were not as
cache-isolated as intended. Their actual compiled-module snapshots remain the
authoritative replay reference, not the cloned save's cache aliases.

Build the external-fixture runner with
`cmake --build engine\out\gpu --target ggx-directional-test --parallel 4`.
`engine\tools\re\ggx_directional_replay.py` requires explicit `--vertex`, `--fragment`
and `--runner` paths and a new output directory. Both supplied shaders must match
their `compiled-<stage-key>.spv` snapshots byte-for-byte. The actual vertex module
is copied without descriptor relinking or arithmetic edits; UBO/SSBO types come
from that module's reflection. The recorded pipeline, depth format, sampler,
viewport, scissor, culling, clipping and blend state are required. Incomplete old
captures are rejected rather than repaired with a guessed depth-format override.
Native mode is the default; `--mode compare` cannot score either GGX candidate
unless both native packed targets match with **zero differing pixels**. Subsequent
unblended comparisons require finite output, byte-exact diffuse preservation
(including signed zeros), unchanged specular blend alpha and a real specular change.
The device must enable `sampleRateShading` for the native shader's interpolation
capability even though this one-sample pass has per-sample shading disabled.

The previous **one diffuse / 31 specular pixel** discrepancy came from replaying
a 9,664-word loose-cache module instead of the actual 7,981-word module used by
the captured draw. Using the actual compiled fragment and vertex plus captured
pipeline state reproduces **both 1280x720 native packed targets exactly**, on two
captured draws, and repeated native replay is byte-identical. The zero-mismatch
threshold has not been relaxed. Both width and peak GGX candidates now run against
that valid baseline with finite output and byte-exact diffuse preservation.
These are offline directional-light comparisons, not a complete game conversion
or a selected brightness calibration. Final composited appearance remains
unevaluated; limited offline GPU timing is described below.
Nothing was installed or enabled in the game to obtain these results.

`engine\tools\re\ggx_directional_controls.py <verified-replay> <new-output-dir>
--runner <ggx-directional-test.exe>` adds offline GPU controls only after checking
the exact scene's source hashes, actual compiled modules, packed targets and both
candidate outputs. It deliberately constructs synthetic inputs around the
unchanged native vertex/fragment interface: five specular-color swatches, all
8-bit smoothness extremes, narrow/broad light widths, diffuse-wrap isolation,
all four specular lobes, optional-lobe switches, opposing directions and a normal
sweep spanning front, grazing and back-facing views. These derived fixtures are
not new captures and must not be substituted for the zero-mismatch scene gate.

Both GGX mappings are finite on these controls, preserve diffuse and specular
blend alpha byte-for-byte, keep disabled/zero-radiance lobes exactly dark, and
suppress back-facing microfacet specular exactly. Zero encoded specular color
still retains native Schlick grazing reflection; it is not a zero-radiance light.
GPU specular is checked against an independent float64 BRDF evaluation with
absolute tolerance `1e-6` plus relative tolerance `0.002` for floating-point
arithmetic, **not** as a relaxation of packed native scene equivalence.

The controls confirm longer GGX highlight tails but do not establish a universally
correct brightness mapping. In the mirrored-grazing control, the width mapping's
peak is approximately **2.8x native**, and the peak mapping's approximately
**4.8x native**. Native optional lobes also have a back-facing response that the
GGX candidates deliberately remove. Neither mapping is selected for production.
The tool writes labeled material/angular sheets and a captured directional-draw
preview using the same fixed gain and Reinhard/sRGB transform across candidates.
The preview excludes final composition, native environment/SSR, exposure and
post-processing, so it is not a final in-game before/after. Inputs and PNGs remain
local and ignored; the tool never launches or installs the game.

The controls' explicit `--calibrate` option compares native, width-matched GGX
and a new offline `ggx-width-authored` hypothesis. The latter keeps the original
effective-exponent/light-width formula and the width-matched distribution, but
multiplies by the analytically derived ratio of native to GGX response at the
mirrored reference orientation (surface normal aligned with the half vector).
The reference includes the native primary-light cosine gate; optional lobes keep
their distinct normalization. This gain depends on the view/light geometry and
effective exponent, not on an arbitrary maximum brightness. It leaves native
Schlick Fresnel, diffuse, visibility inputs and table weights untouched, and
never clips specular output. It is an **authored compatibility response**, not
an energy-conserving or complete PBR conversion. The original two candidates
and native-default generator/replay behavior are unchanged.
The replay tool can also evaluate this candidate explicitly with
`--mode compare --candidate ggx-width-authored`; repeat `--candidate` to compare
several modes. Candidate selection is rejected in native mode and cannot bypass
the exact packed baseline or float diffuse/alpha gates.

The calibration checks all 256 smoothness values against three native light
widths analytically, plus 16 GPU material rows and 58 angular rows. Roughness
alone cannot match both reference brightness and native width: for smoothness
128 and light width 50, the native distribution half-width is 7.08 degrees,
whereas fitting peak brightness only would need approximately 5.49 degrees at
normal incidence and 10.12 degrees at the mirrored 80-degree grazing reference.
The authored normalization preserves the distribution's half-width; it does
not guarantee the full Smith/cosine lobe's maximum or apparent width.

In the sampled mirrored-grazing control, native anchoring reduces the peak ratio
from approximately **2.78x to 1.16x native**. Across the sampled ordinary material
spheres its peaks are approximately 0.97-1.03x native, but the very rough optional
grazing cases still reach approximately **1.93x native**. The captured directional
draw's mean sampled specular rises approximately **48%** because restoring peaks
while retaining long tails is not brightness-neutral. These are measured pass
values, not overall game brightness or appearance claims. Both captured native
packed targets remain exact, and calibrated float comparisons preserve diffuse
and specular alpha byte-for-byte. No production model is selected; final
composited appearance and energy behavior remain unevaluated.
The calibration outputs retain hashes, explicit reference/roughness tradeoffs
and labeled previews locally, without launching or changing the installed game.

`engine\tools\re\ggx_directional_timing.py <capture> <new-output-dir>
--vertex <actual-compiled-vertex.spv> --fragment <actual-compiled-fragment.spv>
--runner <ggx-directional-test.exe>` measures native, width, peak and authored
directional variants with Vulkan timestamps. It first reruns the zero-mismatch
native packed gate and all finite, diffuse-exact and alpha-exact float checks.
Each of three independent trials compiles all four pipelines before 256 warmup
rounds and 1,024 measured paired rounds, rotating execution order every round.
All variants use the same device, captured **packed additive targets**, depth,
resources and native raster state. Targets are restored from the captured
pre-draw contents before every draw, outside the timestamp interval, rather
than accumulating repeated lighting. A full execution dependency precedes the
TOP/BOTTOM timestamps around dynamic rendering, including attachment load/store.
Compilation, uploads/restoration, process launch, CPU waits and readback are not
the reported pass time. Native packed output must still be exact after every
trial. Source captures and compiled modules must remain unchanged.

On the RTX 4060 Ti at 1280x720, after closing a competing game, six trials across
the two captured draws gave native median intervals of approximately
**0.046-0.048 ms**, width **0.048-0.050 ms**, peak **0.048-0.050 ms** and authored
**0.050-0.051 ms**. Median within-round ratios were approximately **+5%**, **+4%**
and **+8%**, respectively. These are provisional median pass-cost estimates:
long-tail samples still reach roughly 0.19-0.20 ms and paired percentile ranges
are wide, so this is not a claim of fully isolated or universally reproducible
GPU cost. The initial measurements with another game running were too noisy
to support small cost differences. No other process is stopped by the tools.

Reports retain every timestamp interval, device/driver identity, timestamp
resolution, per-trial medians, p10/p90 spreads, paired ratios, native drift
diagnostics and fixture/runner hashes. Validation layers should be enabled for
a short correctness run, then disabled for timing; the requested layer
environment is recorded. The serialized, warm-cache captured workload is not
an isolated shader-ALU benchmark or a complete live frame. **No whole-game FPS,
final composited appearance or default production choice is established**, and
nothing is installed or enabled in the game by this benchmark.

#### Opt-in native-anchored directional GGX

The renderer now integrates `ggx-width-authored` as **Directional GGX
(native-anchored, experimental)** under **Game effects (restart required)**.
The saved setting is `graphics_directional_ggx=1`; it defaults to `0`.
Restart Bloodborne after changing it. The startup value is frozen for the
session so newly compiled and warmed-up pipelines cannot select different
models after a menu toggle. Picture tuning, native skin SSS and this switch
remain independent.

`vk_ggx_patch.h` emits only the authored scalar GGX instructions and rewrites
the four guarded directional specular sites. It contains no captured shader
binary. Both supported exact compiled layouts are validated; unknown layouts,
other stages and other shader hashes remain native. An unsupported directional
variant produces a renderer warning rather than attempting a guessed rewrite.
Native material/light-width decoding, Fresnel, diffuse and visibility stay
unchanged. Local lights, forward materials, environment and SSR are **not**
converted. This remains an authored compatibility response, not full PBR.

Cold compilation applies the transformation to the selected module (including
an exact supported external shader override); cached-pipeline warmup uses the
same emitter. **Persistent directional shader binaries remain native**.
Selection is reapplied only to the module submitted to Vulkan, so switching
off and restarting restores native specular without manually deleting caches
or leaving GGX baked into the native fallback.

The runtime-emitted legacy shader reproduces the offline authored result
byte-for-byte on both captured scenes and all 16 material / 58 angular control
rows; native packed targets, diffuse and specular alpha remain unchanged.
`ggx-patch-test` exercises disabled/unrelated/unknown/already-patched gates and
both layouts; `engine\tests\test_ggx_runtime.py` validates SPIR-V and GPU
equivalence using external local fixtures. Settings regressions cover default,
save/reload and frozen restart selection.

Native-anchored GGX is the strongest candidate from the controlled comparisons,
not a verified best final-game look. Its longer tails still raise sampled
mean specular by approximately 48% and very rough grazing peaks can reach
1.93x native. The worktree engine is built with this optional feature. It was subsequently
installed and enabled for a local in-game comparison, with positive visual
feedback; this is subjective feedback, not a broad scene-by-scene assessment. Final composited
appearance, especially rough/grazing surfaces and skin with native SSS, still
needs an in-game comparison before choosing a default.

### Renderer optimizations

Renderer optimizations are permanently enabled, with no menu toggle.
The shadow shader skips projection setup for
sky/invalid depth and skips blocker/filter sampling after native directional
shadows have faded completely. Existing validity checks, partial fading,
cascade blending, contact shadows, sample counts and resolutions are preserved.
It also skips the camera's diagnostic inverse calculation when neither the
alternate-inverse path nor camera logging uses it. Gameplay and audio paths
are unchanged. Older `renderer_optimizations` settings are no longer used.

Benefits depend on the scene; this is not a guaranteed overall FPS increase.
The standalone `directional-shadow-test --benchmark` measures both pipelines with
GPU timestamps on visible and fully faded shadows. Its correctness suite compares
the complete RGBA output of both variants byte-for-byte, including atlas edges,
cascade blends, sloped surfaces, sky and invalid coordinates.

### Higher-resolution native local-light shadows

The game's original shadowcasting point lights use variance shadow maps, storing depth and squared-depth moments on six cube faces. At Very high local shadow resolution, BBFSS raises their native pool's base face size to 2048 pixels, producing **2048/1024/512/256** distance-quality tiers. The native allocator creates the larger faces and shared working targets, and the rendering and filtering passes consume their dimensions. This adds actual shadow silhouette detail rather than upscaling an existing low-resolution map.

The separate **Game effects (restart required) > Local shadow resolution** dropdown
controls point lights and spotlights together. It is saved as
`graphics_local_shadow_resolution=0/1/2`; restart Bloodborne to apply it.

| Quality | Point-light face tiers | Spotlight tiers |
|---|---|---|
| Default (selected by default) | Native allocations, normally 256/128/64/32 | Native allocations, normally 512/256/128/64 |
| High (4x dimensions) | 1024/512/256/128 | 2048/1024/512/256 |
| Very high (8x dimensions) | 2048/1024/512/256 | 4096/2048/1024/512 |

Default leaves the original allocation instructions and filter constants untouched,
including any native profile-dependent sizing. High uses the original 3.2-texel
filter radius; Very high uses 6.4 texels. Both upgraded qualities retain the same
quarter-stock normalized filter footprint. Directional atlas quality and Picture
tuning do not affect this setting.

Native spotlight shadows receive the same eightfold increase per dimension: their original **512/256/128/64** tiers become **4096/2048/1024/512**. The spotlight allocator also derives both dimensions and the shared working targets from its base size.

The native moment filter uses a 6.4-texel tap radius instead of the original 3.2. At eight times the original map resolution, this gives one quarter of the original normalized blur footprint: a tighter filter to preserve silhouette detail without leaving edges completely unfiltered. Its normalized 13-tap weights and light visibility calculations are unchanged. Filter tuning applies only to light types whose resolution upgrade was installed successfully.

Directional shadows and the separate supplemental screen-space point shadows are unchanged. Higher-resolution maps cost more GPU memory and rendering time. The allocation patches require verified 1.09 code; set `BB_NATIVE_LOCAL_SHADOWS=0` for original point-shadow sizes or `BB_NATIVE_SPOT_SHADOWS=0` for original spotlight sizes before launch.

Implementation: the guarded native point- and spotlight-pool patches in [`graphics.cpp`](engine/gpu/shim/engine/graphics.cpp).

### Contact shadows

Short screen-space rays march toward the directional light to find nearby occluders that the shadow map is too coarse to resolve. The result supplements the existing shadow visibility, adding fine grounding around small objects and surface details.

Contact shadows complement PCSS rather than replace shadow maps. They can only detect occluders represented in scene depth, so off-screen or hidden geometry remains the responsibility of the underlying shadow system.

Implementation: the contact-shadow path in [`contact_shadow.comp`](engine/gpu/shadps4/video_core/host_shaders/contact_shadow.comp).

### Bokeh depth of field

Bloodborne's original native Yebis depth of field is used by default. The modern
bokeh replacement below is retained as an optional implementation: set
`BB_MODERN_DOF=1` before launch to enable it. Its added autofocus, foreground blur
and intensity slider do not modify native DOF when the replacement is disabled.

Modern depth-aware bokeh preserves each map's resolved far-DOF settings instead of imposing one global photographic preset. The renderer consumes the game's encoded blur curve and measures the native blur kernel's covariance on the GPU to calibrate a disk filter with equivalent second moment and anisotropy.

Bilateral half-resolution preparation, a disk gather and depth-aware full-resolution reconstruction produce the far blur. Sparse full-resolution edge repair stabilizes thin foliage. Independently authored near blur adds upper-centre autofocus, a robust median depth target and time-based focus smoothing.

A feathered central exemption protects typical close-up head, shoulder and cloak framing from near blur. This is a screen-region heuristic, not player identification: nearby scenery inside it is also protected. The saved **Depth of field intensity** slider scales blur after the map-specific settings, without moving the authored far-focus ranges.

Implementation: [`depth_of_field.comp`](engine/gpu/shadps4/video_core/host_shaders/depth_of_field.comp) and [`vk_depth_of_field.cpp`](engine/gpu/shadps4/video_core/renderer_vulkan/vk_depth_of_field.cpp). See the [DOF implementation notes](engine/docs/depth-of-field.md) for calibration, fallbacks and limitations.

## Upstream installation reference

The downloads below are **official Lantern Engine builds, not BBFSS builds**, and do not include this fork's custom visual changes. They are retained as an upstream setup reference. BBFSS-specific binary distribution is separate from those releases.

1. Download `LanternEngine-<version>-win64.zip` from [Releases](https://github.com/Awakenchan/LanternEngine/releases) and extract it anywhere.
2. Run `Lantern Engine.exe` and pick your game folder (the one with `eboot.bin`).
3. Press **Play**. The launcher updates itself. `Bloodborne.exe` starts the game directly (add it to Steam as a non-Steam game).

### Linux (beta)

1. Download `LanternEngine-<version>-linux-x64.tar.gz` from [Releases](https://github.com/Awakenchan/LanternEngine/releases) and unpack it (`tar xzf LanternEngine-*-linux-x64.tar.gz`).
2. Run `Lantern Engine` in the unpacked folder, pick your game folder and press **Play**. From a terminal: `./lantern.sh /path/to/CUSA03173`.
3. It needs a Vulkan driver (Mesa or NVIDIA), Python 3 and glibc 2.38 or newer (Ubuntu 24.04, Fedora 39, Arch, SteamOS 3.6, Bazzite). From 0.9.37 beta 7 the launcher updates itself on Linux too, and DLSS Super Resolution works on RTX cards with NVIDIA's driver (DLSS Frame Generation is Windows only).

## Credits

| | |
|---|---|
| **BBFSS** | Unofficial modified fork maintained by [fromsoftserve](https://github.com/fromsoftserve), adding the renderer enhancements described above |
| **Lantern Engine** | by **Awaken**: the launcher, the built-in game patches, and the engine reworked from its starting point (native shader compiler, direct draw calls, draw pipeline and memory performance work, PC menus and controls, upscaler integration, cheats, occlusion culling, trophies, fixes) |
| Engine starting point | [DarkIzuku/bloodborne_pc](https://github.com/DarkIzuku/bloodborne_pc) (built on [bbport](https://github.com/deadinside28/bloodborne_pc) and [shadPS4](https://github.com/shadps4-emu/shadPS4)) |
| Renderer crash fixes | adapted from [AwesomeObserver](https://github.com/AwesomeObserver/bloodborne_pc) and [deadinside28](https://github.com/deadinside28/bloodborne_pc) |
| Online backend for The Hunter's Dream | [droogie/bbhost](https://github.com/droogie/bbhost) |
| Online servers | [The Hunter's Dream](https://thehuntersdream.com) |
| FSR 3.1 and FSR 4 on Vulkan | [FireBurn/FSR-Vulkan](https://github.com/FireBurn/FSR-Vulkan) and the AMD FidelityFX SDK |
| FSR 4.1.1 | [fsr4vk](https://github.com/dvj5411/fsr4vk), with the AMD driver fix from [0xCydral](https://github.com/0xCydral/bloodborne_pc) |
| DLSS | NVIDIA DLSS SDK |
| Sound | [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9) |
| In-game menu | [Dear ImGui](https://github.com/ocornut/imgui) |
| Controller mappings | [SDL_GameControllerDB](https://github.com/mdqinc/SDL_GameControllerDB) |
| UI library | [WPF-UI](https://github.com/lepoco/wpfui) |
| GTAO slice formulation | Jimenez et al., *Practical Real-Time Strategies for Accurate Indirect Occlusion* (Activision TR-16-01) |
| Visibility-bitmask AO and SSGI | Therrien, Levesque and Gilet, [*Screen Space Indirect Lighting with Visibility Bitmask*](https://arxiv.org/abs/2301.11376) |
| PCSS technique | NVIDIA's Percentage-Closer Soft Shadows |

## License

Lantern Engine's own code, the launcher and the built-in game patches are GPL-3.0-or-later with additional terms (see [LICENSE](LICENSE) and [NOTICE](NOTICE)): anything that uses or ships them must keep the credit "Lantern Engine by Awaken", changed versions must say they are changed and must not pass as an official Lantern Engine release, and the name "Lantern Engine" is not licensed for other projects. Parts by others keep their own licenses, listed in the engine's `licenses/` folder.

Source: the engine's full source is in [`engine/`](engine) (0.9.37 beta 7 on), the launchers' in [`launcher/windows/`](launcher/windows), and the Help & FAQ page in [`faq/`](faq).
