/**
 * Per-agent garment recolouring, done on the GPU as the sprite is drawn.
 *
 * Nothing is generated: the four source atlases are the only textures ever loaded, whatever the
 * size of the cast; the fragment shader rewrites each pixel's colour on its way to the screen.
 *
 * The colour is the agent's identity colour (SoulLayer.color), authored for this world's theme and
 * made distinct across the cast by world/identity_color.py, so the renderer hardcodes no palette.
 * It matches the nameplate and ground ring, so a figure reads as one identity.
 *
 * The garment takes the identity colour's hue, saturation AND value: identity_color separates the
 * cast in the full colour space, and its two near-neutral slates (#8a94a8, #6a7690) sit at hue 220°
 * and 221°, so hue alone would give them the same shirt. The art contributes each pixel's relative
 * value (folds, shading): a garment pixel is re-lit as the identity colour scaled by how light the
 * artist painted it.
 *
 * Hue and saturation cannot separate hair from skin in this art (#bf7958 is the hair of the
 * pale-skinned bodies and the skin of the dark-skinned ones), so warm is off limits:
 *
 *   1. v ≈ 0            → outline. Keep.
 *   2. hue ≤ 60°        → skin (any tone, pale #ffd7b1 through dark #bf7958), brown hair,
 *                         leather and, since a neutral's hue reads as 0, the whites of the
 *                         eyes. Keep.
 *   3. otherwise        → garment (coloured shirts, and neutral dark clothing / black hair /
 *                         white shirts alike). Re-lit as described above.
 *
 * Premultiplied alpha needs no special handling: hue and saturation are invariant under the
 * uniform rgb scaling it applies, and value scales with it.
 */

import Phaser from "phaser";

import { ART_LIT, LINE_VAL, WARM_MAX } from "../lib/dye";

/** A GLSL float literal: always with a decimal point, which GLSL ES requires. */
const glsl = (n: number): string => n.toFixed(6);

const FRAG = `
#define SHADER_NAME ASAMANA_RECOLOR_FS
precision mediump float;

uniform sampler2D uMainSampler;
uniform vec3 uDye;         // the identity colour as HSV — hue, saturation, value

varying vec2 outTexCoord;

const float WARM_MAX = ${glsl(WARM_MAX)};  // skin / brown hair / leather / neutral-white
const float LINE_VAL = ${glsl(LINE_VAL)};  // near-black outlines
// The value the artist painted a lit garment at. A pixel's value RELATIVE to this is what
// carries the folds, so we re-light with (pixelValue / ART_LIT) × the identity's value:
// a mid-tone lands exactly on the identity colour, highlights and shadows fall either side.
const float ART_LIT  = ${glsl(ART_LIT)};

vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
  vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
  float d = q.x - min(q.w, q.y);
  float e = 1.0e-10;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
  vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
  vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
  return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
  vec4 tex = texture2D(uMainSampler, outTexCoord);
  if (tex.a == 0.0) { gl_FragColor = tex; return; }

  vec3 hsv = rgb2hsv(tex.rgb);
  if (hsv.z <= LINE_VAL || hsv.x <= WARM_MAX) { gl_FragColor = tex; return; }

  float lit = clamp(hsv.z / ART_LIT * uDye.z, 0.0, 1.0);
  gl_FragColor = vec4(hsv2rgb(vec3(uDye.x, uDye.y, lit)), tex.a);
}
`;

export const RECOLOR_FX = "AsamanaRecolor";

export class RecolorFX extends Phaser.Renderer.WebGL.Pipelines.PostFXPipeline {
  /** The identity colour as HSV (each 0..1). Set once per token; never changes. */
  dye: [number, number, number] = [0, 0, 1];

  constructor(game: Phaser.Game) {
    super({ game, fragShader: FRAG, name: RECOLOR_FX });
  }

  onPreRender(): void {
    this.set3f("uDye", this.dye[0], this.dye[1], this.dye[2]);
  }
}
