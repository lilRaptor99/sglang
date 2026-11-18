# SGLang MoE expert-activation handoff

## Status as of November 19, 2026

The Spartan H100 BF16 sweep is complete. Scratch contains 20 production
`expert_counts.json` files covering MMLU, BBH, HumanEval, INCLUDE, and PopQA
for each model. The old single-dash OLMoE smoke output is absent.

| Model | MoE capture dimensions (layers × experts, top-k) | Final Slurm job |
|---|---|---|
| OLMoE-1B-7B-0924-Instruct | 16 × 64, 8 | 33009563 |
| DeepSeek-MoE-16B-Chat | 28 × 64, 6 | 33009563 |
| Qwen3-30B-A3B | 48 × 128, 8 | 32969746 |
| Mixtral-8x22B-Instruct-v0.1 | 56 × 8, 2 | 33416348 |

Qwen ran October 15, and job 33009563 regenerated DeepSeek and OLMoE after
the stride fix. The first Mixtral TP=4 attempt (33139440) exited with Slurm
status COMPLETED but produced zero files because `google.protobuf` was
missing at tokenizer startup. The four-H100 rerun (33416348) finished
October 20 in 1h37m. All five Mixtral JSONs were parsed and checked for
architecture, token totals, and aggregate fields. The other 15 files were
schema-checked during the original sweep and their presence was rechecked.

Results are under
`/data/scratch/projects/uom00014/vllm/results_sglang_bf16_h100/<org>--<model>/bf16/moe-<benchmark>/expert_counts.json`.
The shared schema carries model architecture, totals, marginal and pair
counts, and per-subject layer counts.

## Reproduce on Spartan

The evaluation code lives in [vLLM examples/eval-moe](../vllm/examples/eval-moe).
Use `moe_expert_stats_sglang.py` and the scripts in its `spartan/` directory.
`sglang-moe-eval-h100-bf16-wide.sbatch` uses one H100 for the smaller models.
`sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch` uses four H100s for Mixtral.
Verify actual output files and driver logs after every run. Slurm's
COMPLETED status can mask a driver or server failure.

- Known working venv on scratch is `vllm/venv_sglang_0509`. Pins are
  SGLang 0.5.9, sgl-kernel 0.3.21, torch 2.9.1+cu128, flashinfer-python
  0.6.7.post1, NumPy 2.2.6, datasets 4.5.0, PyArrow 26.0.0, and httpx
  0.28.1. Check protobuf on rebuild. Newer SGLang versions tested pulled
  CUDA 13 dependencies while Spartan was on CUDA 12.8.
- Reapply the installed `sglang/srt/layers/moe/routed_experts_capturer.py`
  layer-ID patch on rebuild. CUDA-graph capture can supply `layer_id=None`.
  The patch synthesizes monotonic layer IDs in that case. Look for the
  `synthesize` marker in the working venv. The launch scripts enable
  `ServerArgs.enable_return_routed_experts` through a spawn-safe wrapper
  because 0.5.9 lacks the CLI flag.
- The driver uses raw HTTP via httpx to retain `sglext.routed_experts`.
  The OpenAI SDK parsing path dropped it. Commit `89a733cd7` fixed the
  wide-sweep tuple handling. Commit `2ed502b4e` fixed DeepSeek decoding:
  try `top_k=6` first, with the shared-expert-inclusive stride as fallback.
- For a rebuild smoke test, run OLMoE/PopQA with 8 prompts into a separate
  output tree. Check that expert counts and per-layer data are populated
  before submitting the full sweep.

## Open work

Run or locate the three-way BF16 Jaccard comparison with vLLM and llama.cpp.
No completed SGLang-inclusive report was found in the locations checked.
The [main evaluation handoff](../vllm/EVAL_MOE_HANDOFF.md) tracks the
comparison inputs. An SGLang MediaFlux upload was not verified. TensorRT-LLM
Router Replay was separately blocked by CUDA 13 dependencies on Spartan.
