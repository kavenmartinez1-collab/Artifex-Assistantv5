/**
 * Vision tensor locator — canonical roles → tensor names per weight source.
 *
 * The vision twin of TensorLocator. Two naming worlds:
 *   - HF safetensors (Qwen3-VL ground truth, fused QKV, biases):
 *       model.visual.patch_embed.proj.weight, model.visual.pos_embed.weight,
 *       model.visual.blocks.{i}.{attn.qkv,attn.proj,norm1,norm2,
 *       mlp.linear_fc1,mlp.linear_fc2}.{weight,bias},
 *       model.visual.merger.{norm,linear_fc1,linear_fc2}.{weight,bias},
 *       model.visual.deepstack_merger_list.{d}.{norm,linear_fc1,linear_fc2}.*
 *   - GGUF clip.cpp (mmproj files / Ollama-packed v.* tensors, separate QKV):
 *       v.patch_embd.weight, v.position_embd.weight,
 *       v.blk.{i}.{attn_q,attn_k,attn_v,attn_out,ln1,ln2,ffn_up,ffn_down}.*,
 *       mm.* projector tensors (per-family suffixes).
 *
 * locate() returns null for roles a source/family doesn't have — the runner
 * decides which roles are required (e.g. qkv vs q/k/v based on
 * descriptor.fusedQKV).
 */

export type VisionRole =
  | 'patchEmbed' | 'patchEmbedBias'
  | 'posEmbed'
  | 'postNorm' | 'postNormBias'
  // per-block (blockIdx required)
  | 'qkv' | 'qkvBias'
  | 'q' | 'qBias' | 'k' | 'kBias' | 'v' | 'vBias'
  | 'attnOut' | 'attnOutBias'
  | 'norm1' | 'norm1Bias' | 'norm2' | 'norm2Bias'
  | 'mlpFc1' | 'mlpFc1Bias' | 'mlpFc2' | 'mlpFc2Bias'
  // projector
  | 'mergerNorm' | 'mergerNormBias'
  | 'mergerFc1' | 'mergerFc1Bias' | 'mergerFc2' | 'mergerFc2Bias'
  // deepstack (dsIdx required)
  | 'dsNorm' | 'dsNormBias' | 'dsFc1' | 'dsFc1Bias' | 'dsFc2' | 'dsFc2Bias';

export interface VisionLocator {
  /** Resolve a role to a tensor name, or null if absent in this checkpoint. */
  locate(role: VisionRole, blockIdx?: number, dsIdx?: number): string | null;
  /** All vision tensor names present (for load planning). */
  readonly names: string[];
}

type Pattern = string; // {i} = block index, {d} = deepstack index

const HF_QWEN_PATTERNS: Record<VisionRole, Pattern | null> = {
  patchEmbed: 'model.visual.patch_embed.proj.weight',
  patchEmbedBias: 'model.visual.patch_embed.proj.bias',
  posEmbed: 'model.visual.pos_embed.weight',
  postNorm: null,
  postNormBias: null,
  qkv: 'model.visual.blocks.{i}.attn.qkv.weight',
  qkvBias: 'model.visual.blocks.{i}.attn.qkv.bias',
  q: null, qBias: null, k: null, kBias: null, v: null, vBias: null,
  attnOut: 'model.visual.blocks.{i}.attn.proj.weight',
  attnOutBias: 'model.visual.blocks.{i}.attn.proj.bias',
  norm1: 'model.visual.blocks.{i}.norm1.weight',
  norm1Bias: 'model.visual.blocks.{i}.norm1.bias',
  norm2: 'model.visual.blocks.{i}.norm2.weight',
  norm2Bias: 'model.visual.blocks.{i}.norm2.bias',
  mlpFc1: 'model.visual.blocks.{i}.mlp.linear_fc1.weight',
  mlpFc1Bias: 'model.visual.blocks.{i}.mlp.linear_fc1.bias',
  mlpFc2: 'model.visual.blocks.{i}.mlp.linear_fc2.weight',
  mlpFc2Bias: 'model.visual.blocks.{i}.mlp.linear_fc2.bias',
  mergerNorm: 'model.visual.merger.norm.weight',
  mergerNormBias: 'model.visual.merger.norm.bias',
  mergerFc1: 'model.visual.merger.linear_fc1.weight',
  mergerFc1Bias: 'model.visual.merger.linear_fc1.bias',
  mergerFc2: 'model.visual.merger.linear_fc2.weight',
  mergerFc2Bias: 'model.visual.merger.linear_fc2.bias',
  dsNorm: 'model.visual.deepstack_merger_list.{d}.norm.weight',
  dsNormBias: 'model.visual.deepstack_merger_list.{d}.norm.bias',
  dsFc1: 'model.visual.deepstack_merger_list.{d}.linear_fc1.weight',
  dsFc1Bias: 'model.visual.deepstack_merger_list.{d}.linear_fc1.bias',
  dsFc2: 'model.visual.deepstack_merger_list.{d}.linear_fc2.weight',
  dsFc2Bias: 'model.visual.deepstack_merger_list.{d}.linear_fc2.bias',
};

/** GGUF clip.cpp conventions — projector names vary per family; the common
 *  v.* tower names are stable. Verified against real mmproj files in M2. */
const GGUF_CLIP_PATTERNS: Record<VisionRole, Pattern | null> = {
  patchEmbed: 'v.patch_embd.weight',
  patchEmbedBias: 'v.patch_embd.bias',
  posEmbed: 'v.position_embd.weight',
  postNorm: 'v.post_ln.weight',
  postNormBias: 'v.post_ln.bias',
  qkv: null,
  qkvBias: null,
  q: 'v.blk.{i}.attn_q.weight', qBias: 'v.blk.{i}.attn_q.bias',
  k: 'v.blk.{i}.attn_k.weight', kBias: 'v.blk.{i}.attn_k.bias',
  v: 'v.blk.{i}.attn_v.weight', vBias: 'v.blk.{i}.attn_v.bias',
  attnOut: 'v.blk.{i}.attn_out.weight',
  attnOutBias: 'v.blk.{i}.attn_out.bias',
  norm1: 'v.blk.{i}.ln1.weight',
  norm1Bias: 'v.blk.{i}.ln1.bias',
  norm2: 'v.blk.{i}.ln2.weight',
  norm2Bias: 'v.blk.{i}.ln2.bias',
  mlpFc1: 'v.blk.{i}.ffn_up.weight',
  mlpFc1Bias: 'v.blk.{i}.ffn_up.bias',
  mlpFc2: 'v.blk.{i}.ffn_down.weight',
  mlpFc2Bias: 'v.blk.{i}.ffn_down.bias',
  mergerNorm: 'mm.norm.weight',
  mergerNormBias: 'mm.norm.bias',
  mergerFc1: 'mm.fc1.weight',
  mergerFc1Bias: 'mm.fc1.bias',
  mergerFc2: 'mm.fc2.weight',
  mergerFc2Bias: 'mm.fc2.bias',
  dsNorm: null, dsNormBias: null,
  dsFc1: null, dsFc1Bias: null,
  dsFc2: null, dsFc2Bias: null,
};

export function createVisionLocator(
  tensorNames: Iterable<string>,
  source: 'hf' | 'gguf',
): VisionLocator {
  const present = new Set(tensorNames);
  const patterns = source === 'hf' ? HF_QWEN_PATTERNS : GGUF_CLIP_PATTERNS;

  const visionNames = [...present].filter(n =>
    n.startsWith('model.visual.') || n.startsWith('visual.')
    || n.startsWith('v.') || n.startsWith('mm.'));

  return {
    names: visionNames,
    locate(role: VisionRole, blockIdx?: number, dsIdx?: number): string | null {
      const pattern = patterns[role];
      if (!pattern) return null;
      const name = pattern
        .replace('{i}', String(blockIdx ?? 0))
        .replace('{d}', String(dsIdx ?? 0));
      return present.has(name) ? name : null;
    },
  };
}
