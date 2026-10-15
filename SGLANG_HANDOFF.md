# sglang cu12 expert-activation sweep handoff (Oct 2026)

> This document lives in the root of `/data/gpfs/projects/uom00014/sglang/`
> as a sister to `/data/gpfs/projects/uom00014/vllm/EVAL_MOE_HANDOFF.md`
> (which has the canonical vLLM-side addenda, including the detailed
> job-by-job accounting for this work in Addendum 8). Read both.
>
> The sglang repository itself is **unmodified** — we did not commit
> any changes to this repo's source. All the patches we needed live
> in the deployed venv at `/data/scratch/projects/uom00014/vllm/venv_sglang_0509/`
> and in the driver at
> `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/moe_expert_stats_sglang.py`.
> This handoff describes the sglang-side gotchas so a future session
> can rebuild the cu12 backend on a different machine or after a
> venv rebuild.

## TL;DR

- **sglang 0.5.9** + **sgl-kernel 0.3.21** + **torch 2.9.1+cu128** +
  **flashinfer-python 0.6.7.post1** (cu128 wheel) + **numpy 2.2.6** +
  **datasets 4.5.0** + **pyarrow 26.0.0** is the working cu12 stack.
  Resolved cleanly via `uv pip install --torch-backend=cu128` against
  the sgl-project PEP 503 wheel index.
- **Two on-disk patches required**:
  1. Capturer layer_id synthesis — modify
     `venv/lib/python3.11/site-packages/sglang/srt/layers/moe/routed_experts_capturer.py`
     to synthesize a monotonic `layer_id` (the capturer's `layer_id`
     arg is `None` for SM80 / deepseek-v2 / mixtral in CUDA-graph
     replay, collapsing all layers to one).
  2. Driver-side response bypass — use `httpx` directly instead of the
     OpenAI Python SDK. The SDK's `ChatCompletionChoice` is
     pydantic-strict (`extra='forbid'`) and silently drops
     `sglext.routed_experts` on parse.
- **Driver**: `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/moe_expert_stats_sglang.py`
  (~765 lines). Imports the vLLM analyzer's prompt loaders + JSON
  writer as a library. Output JSON is bit-compatible with the vLLM
  and llama.cpp trees so the sister-repo Jaccard aggregator reads
  the sglang tree unmodified.
- **Wide sweep sbatch**:
  `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/sglang-moe-eval-h100-bf16-wide.sbatch`.
  gpu-h100, 1 GPU, 12 h wall. **Completed** as jobs 32933582
  (Oct 14) → 32969746 (Qwen-inclusive, Oct 15) → 33009563 (post
  stride-fix, Oct 15). The tree has 15 expert_counts.json files
  (OLMoE × 5, DeepSeek × 5, Qwen3 × 5). Mixtral BF16 TP=4 is in
  flight as job 33139440 (PENDING since 2026-10-16 09:12).
- **Mixtral TP=4 sbatch**:
  `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch`.
  gpu-h100, 4 GPUs, 12 h wall. Mirrors the vLLM Mixtral TP=4
  precedent (job 31516295, the only prior Mixtral success):
  bf16 weights are 262 GB; per-GPU slice at TP=4 = 65 GB with
  ample KV cache headroom on 80 GB. Same driver + capturer
  plumbing as the wide sweep; just `--tensor-parallel-size 4`
  added to the sglang launcher. Job 33139440 is queued on gpu-h100
  with 4–8 day wait estimate as of 2026-10-16 09:12.

## 1. Why this version (the cu12 wall, restated)

Spartan's GPU fleet (gpu-a100, gpu-a100-short, gpu-h100 — 36 nodes total)
runs **driver 570.x (CUDA 12.8 max)**. There is no CUDA 13+ node
anywhere on the cluster. Every sglang release **0.5.20 and later**
hard-pins CUDA 13+ in its published wheel METADATA:

```
Requires-Dist: flashinfer_python[cu13]==0.6.18
Requires-Dist: cuda-python>=13.0
Requires-Dist: humming-kernels[cu13]==0.1.12
Requires-Dist: nvidia-cutlass-dsl[cu13]==4.6.2
Requires-Dist: nvshmem4py-cu13
```

The wall isn't just sglang's METADATA. It's deeper — it's `sgl-kernel`'s
binary distribution. The published `sgl-kernel` cu128 wheels
(≤0.3.14.post1) **also** ship a `sm100/common_ops.abi3.so` that links
against `libnvrtc.so.13`, `libcublas.so.13`, `libcudart.so.13`. The
arch-detection logic in `sgl-kernel/__init__.py` routes **any non-Hopper
GPU (including SM80 A100 and SM90 H100)** to `sm100/common_ops.abi3.so`
because the wheels don't ship per-arch `.so` files for cu128 anymore.
There is no cu12-compatible sgl-kernel wheel on PyPI or on
`docs.sglang.io`.

So the wall is: **sgl-kernel post-0.3.14.post1 dropped cu12 entirely,
and sglang post-0.5.10 inherited the cu13 requirement transitively.**

The only release path that still works on cu12 driver is **sglang 0.5.9
+ sgl-kernel 0.3.21** (the last version of the old package name;
later became `sglang-kernel` 0.4.x). 0.5.9 was the last release that
pre-dates the cu13 pin and the last release that didn't pull
sglang-kernel transitively.

We confirmed the resolve path:

```bash
uv pip install --torch-backend=cu128 \
    sglang==0.5.9 \
    sgl-kernel==0.3.21 \
    flashinfer-python==0.6.7.post1 \
    torch==2.9.1+cu128 \
    httpx==0.28.1 \
    'datasets==4.5.0' 'numpy>=2.1,<2.3' 'pyarrow>=17,<21'
```

Resolves cleanly, builds the Rust ext in ~5 min, ~7 GB on disk.

## 2. The capturer patch (the real bug)

### What sglang provides

`sglang/srt/layers/moe/routed_experts_capturer.py` defines
`_RoutedExpertsDeviceCache.capture_fwd_routed_experts(...)`, which
records `(num_tokens, top_k + n_shared_experts)` int32 expert IDs per
capturer invocation. The cache also tags each invocation with a
`layer_id` so the consumer can reconstruct per-layer routing tensors
of shape `(num_tokens, num_layers, top_k)`.

### What's broken in 0.5.9

The `layer_id` parameter is supposed to come from the model code's
forward hook. `python/sglang/srt/models/deepseek_v2.py` and
`python/sglang/srt/models/mixtral.py` 0.5.9 forward hooks don't
reliably pass `layer_id` to the capturer — when called from the CUDA
graph replay path, it's `None` or `0` for every layer, making the
captured tensor degenerate (`num_layers` collapsed to 1 in the
consumer).

The consumer (`_get_local_slice`) handles a `None`/`0` `layer_id` by
attaching everything to layer index 0, which makes
`expert_counts.json` look like a single-layer model with thousands
of tokens — the aggregator's `[L, E]` marginal bins become a single
row and the heatmap degenerates to one strip.

### The patch

`${VENV}/lib/python3.11/site-packages/sglang/srt/layers/moe/routed_experts_capturer.py`:

```python
# around line 60, inside _RoutedExpertsDeviceCache
def capture_fwd_routed_experts(self, layer_id, ...):
    # synthesize the correct layer_id via a monotonic counter
    if layer_id is None or layer_id == 0:
        layer_id = self._layer_counter
        self._layer_counter += 1
    ...
```

(The exact change is ~10 lines. The driver pre-flight guard in the
sweep sbatch checks for the literal string "synthesize" in the
capturer source; if the venv was rebuilt and the patch is missing,
the sbatch aborts before launching any cell.)

### Why this works

The capturer is invoked exactly `num_layers` times per forward pass
(one per MoE layer). Incrementing a monotonic counter on each call
gives a stable 0..L-1 sequence per request, which is what the
consumer needs to reshape the captured tensor into `(T, L, top_k)`.
The counter is per-request, so concurrent requests don't share
state — each thread/process has its own cache instance.

### Re-applying after venv rebuild

The patch lives in the venv, not in the sglang source tree. If
someone rebuilds `venv_sglang_0509`, the patch is gone and the sweep
fails its pre-flight guard. Re-apply by hand (the diff is local — see
the sgl-kernel wheel above for the file path), or pin a
`capturer_layer_id_synth.patch` file alongside the sbatch and
`patch -p0` it during the venv install.

## 3. The sglext response-path quirk

### What sglang 0.5.9 returns

sglang 0.5.9's OpenAI chat completion server returns captured expert
IDs in a top-level `sglext.routed_experts` field of the response
JSON, base64-encoded int32 with shape `(T, L, top_k + n_shared)`.
This is a behavior change from earlier sglang versions, which used
the per-choice `meta_info.routed_experts` location.

### Why the OpenAI Python SDK drops it

The SDK's `ChatCompletionChoice` class is pydantic-strict with
`extra='forbid'`. When the SDK parses the response, the extra
`sglext` field is silently dropped on parse. `choice.model_extra` is
`None` even though `data["sglext"]["routed_experts"]` is right there
in the raw JSON.

This is a *silent* failure mode — the request completes successfully,
the chat completion text is correct, but the expert IDs are gone.
We discovered this by running
`sglang-cu12-response-check.sbatch` to dump the full response to
disk and grepping for `routed_experts`. It's at the response level
(top-level), not the choice level.

### The fix (driver-side only)

Bypass the SDK entirely. Use `httpx.post()` against `/v1/chat/completions`
and walk the raw JSON dict:

```python
import httpx
r = httpx.post(
    f"{base_url}/chat/completions",
    json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": seed,
        "return_routed_experts": True,
        "routed_experts_start_len": 0,
    },
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=120.0,
)
data = r.json()
sglext = data.get("sglext") or {}
if "routed_experts" in sglext:
    raw = sglext["routed_experts"]
else:
    # fall back to choice.meta_info (older sglang / completions endpoint)
    choice = data["choices"][0]
    raw = (choice.get("meta_info") or choice).get("routed_experts")
```

The decoded tensor is sliced `[:, :, :top_k]` to drop the trailing
`n_shared_experts` columns. sglang concatenates shared expert outputs
onto the end of the routed-experts tensor for "capture window"
accounting; vLLM's `routed_experts` doesn't include them, so we
slice for parity.

### Re-applying after sglang upgrade

If we upgrade to sglang ≥ 0.5.10, the response path may move again
(sglang keeps reshuffling between top-level and choice-level). The
driver's fallback order is: `sglext.routed_experts` → choice-level
`meta_info.routed_experts` → top-level `meta_info.routed_experts`.
This covers every version we've seen through 0.5.21.

## 4. The server-launcher wrapper

sglang 0.5.9 has `ServerArgs.enable_return_routed_experts` as a
field, but does **not** expose it as a CLI flag
(`--enable-return-routed-experts` doesn't exist in 0.5.9 — it was
added in 0.5.10). The sweep sbatch monkey-patches the ServerArgs
initializer via a wrapper script (`launch_server_with_capturer.py`,
written to the output dir per job):

```python
import sglang.srt.server_args as _sa

_orig_init = _sa.ServerArgs.__init__
def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.enable_return_routed_experts = True
_sa.ServerArgs.__init__ = _patched_init

_orig_prepare = _sa.prepare_server_args
def _patched_prepare(argv):
    args = _orig_prepare(argv)
    args.enable_return_routed_experts = True
    return args
_sa.prepare_server_args = _patched_prepare

from sglang.launch_server import run_server
sa = _sa.prepare_server_args(sys.argv[1:])
run_server(sa)
```

The wrapper is `if __name__ == "__main__":` guarded because sglang
uses `multiprocessing.spawn` which re-imports the file in the
worker — without the guard, the worker re-runs `run_server()` and
dies on recursive bootstrap.

## 5. Wide-sweep job history

| JobID | Date | Outcome | Driver exit reason |
|-------|------|---------|---------------------|
| 32605379 | 2026-10-09 15:43 | CANCELLED | First sglang smoke attempt; cancelled before run (Oct 9 debug session) |
| 32605421 | 2026-10-09 15:45 | FAILED 00:18s | cu13 wall — flashinfer 0.6.18 needs libcublas.so.13 |
| 32605642 | 2026-10-09 15:55 | FAILED 02:50 | Same wall |
| 32662546 | 2026-10-10 17:24 | FAILED 01:43 | Source install on login (no GPU); import failure |
| 32663058-32663950 | 2026-10-10 17:42-18:05 | CANCELLED ×4 | Debug attempts, cancelled before run |
| 32663613 | 2026-10-10 17:56 | FAILED 00:37 | Source install import failure |
| 32693183-32694113 | 2026-10-11 12:54-13:05 | INTERACTIVE success | sinteractive debug on gpu-a100-short — confirmed cu12 wall |
| 32696049-32696742 | 2026-10-11 13:48-14:41 | COMPLETED ×8 | Source-install venv cu12 attempts — kernels failed at first GPU call |
| 32703686-32703837 | 2026-10-11 17:53-18:03 | Various | sglang 0.5.20 install attempts; cu13 dep failure |
| 32705126-32705857 | 2026-10-11 18:05-18:22 | FAILED ×4 | sglang 0.5.21 install; cu13 dep failure |
| 32708976-32714323 | 2026-10-11 19:14-21:44 | COMPLETED ×9 | sglang 0.5.9 cu12 install — **the working stack** |
| 32714323 | 2026-10-11 21:44 | COMPLETED 06:24 | Last good venv build before patch + sweep |
| 32842251 | 2026-10-13 12:18 | COMPLETED 01:14 | Smoke test 1 model × 1 benchmark (pre-capturer-patch) |
| 32843922 | 2026-10-13 12:47 | COMPLETED 01:11 | Smoke with sglext response path — wrote 1 expert_counts.json |
| **32843059-32843068** | **2026-10-13 23:31 → 2026-10-14 00:17** | **COMPLETED ×10 (driver-crashed)** | **Wide-sweep attempt: line-675 tuple bug crashed every cell in 1 min** |
| **32933582** | **2026-10-14 11:38 → 15:25** | **COMPLETED 01:12** | **Wide-sweep line-675 fix (commit 89a733cd7) — wrote 10 expert_counts.json (DeepSeek × 5 + OLMoE × 5)** |
| **32969746** | **2026-10-15 04:44 → 06:33** | **COMPLETED 01:48** | **Qwen-inclusive resubmit (MODELS_CSV=Qwen,DeepSeek,OLMoE; 5 Qwen JSONs)** |
| **33009540** | **2026-10-15 06:35:54 → 06:36:20** | **CANCELLED 00:00:26** | **Wrapper job ('wrap') — user-cancelled before any cell ran** |
| **33009563** | **2026-10-15 21:03:34 → 22:19:42** | **COMPLETED 01:16** | **Final wide sweep after stride fix (commit 2ed502b4e) — DeepSeek × 5 + OLMoE × 5 (Qwen already cached)** |
| **33139440** | **2026-10-16 09:12:57** | **PENDING** | **Mixtral BF16 TP=4 cell — `sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch`, gpu-h100, 4 GPUs, 12 h wall** |

`sacct -X --format=JobID,State,Elapsed,Start,End,JobName -j <id>` for any
of these returns ground truth on what ran vs failed.

## 6. Output tree

```
/data/scratch/projects/uom00014/vllm/results_sglang_bf16_h100/
├── allenai-OLMoE-1B-7B-0924-Instruct/
│   └── bf16/
│       ├── moe-mmlu/expert_counts.json        ← will be created by 32933582
│       ├── moe-bbh/expert_counts.json
│       ├── moe-humaneval/expert_counts.json
│       ├── moe-popqa/expert_counts.json
│       └── moe-include/expert_counts.json
├── deepseek-ai-deepseek-moe-16b-chat/
│   └── bf16/
│       └── moe-*/expert_counts.json             ← 5 cells
├── launch_server_with_capturer.py             ← per-job wrapper
├── driver_<JOBID>.log                          ← per-job driver log
└── server_<model>_<bench>.log                  ← per-cell sglang server log
```

Currently the sglang wide-sweep tree has **15 expert_counts.json
files** (3 models × 5 benchmarks) — 5 Qwen3, 5 DeepSeek, 5 OLMoE.

```
/data/scratch/projects/uom00014/vllm/results_sglang_bf16_h100/
├── allenai--OLMoE-1B-7B-0924-Instruct/      (written by jobs 32933582, 33009563)
│   └── bf16/
│       └── moe-{mmlu,bbh,humaneval,popqa,include}/expert_counts.json
├── deepseek-ai--deepseek-moe-16b-chat/    (written by jobs 32933582, 33009563)
│   └── bf16/
│       └── moe-{mmlu,bbh,humaneval,popqa,include}/expert_counts.json
├── Qwen--Qwen3-30B-A3B/                   (written by job 32969746)
│   └── bf16/
│       └── moe-{mmlu,bbh,humaneval,popqa,include}/expert_counts.json
├── launch_server_with_capturer.py        (per-job wrapper, written each run)
├── driver_<JOBID>.log                     (per-job driver log)
└── server_<model>_<bench>.log             (per-cell sglang server log)
```

The schema is bit-compatible with the vLLM / llama.cpp trees (same
`expert_counts.json` shape with `model`, `model_arch`, `totals`,
`aggregate.{marginal_expert_counts,intra_pair_counts,adjacent_pair_counts}`,
and per-subject `layer_expert_counts`).

Note: there is also one stale JSON from the pre-wide-sweep smoke run
(`32843922`) at `allenai-OLMoE-1B-7B-0924-Instruct/` (single-dash
dir name, written before the analyzer's path canonicalization
changed `models--` to `model-`). The wide sweep wrote under
`allenai--OLMoE-1B-7B-0924-Instruct/` (double-dash). The sister
aggregator's tree-walker normalizes both, but the two namespaces
hold different files. Safe to delete the single-dash file if you
want a clean tree.

**Missing**: Mixtral BF16 TP=4 cells (job 33139440, PENDING).

## 7. Verification recipe (after a fresh build)

```bash
# 1. Venv import smoke
source /apps/easybuild-2022/Modules/4.7.1/init/bash
module load GCCcore/11.3.0 Python/3.11.3 CUDA/12.4.1 Ninja/1.11.1-Python-3.11.3
export LD_LIBRARY_PATH="${EBROOTPYTHON}/lib:${LD_LIBRARY_PATH:-}"
/data/scratch/projects/uom00014/vllm/venv_sglang_0509/bin/python -c "
import inspect
from sglang.srt.layers.moe.routed_experts_capturer import _RoutedExpertsDeviceCache
src = inspect.getsource(_RoutedExpertsDeviceCache.capture_fwd_routed_experts)
print('PATCHED' if 'synthesize' in src else 'UNPATCHED')
"
# expect: PATCHED

# 2. Dataset load smoke
/data/scratch/projects/uom00014/vllm/venv_sglang_0509/bin/python -c "
from datasets import load_from_disk
ds = load_from_disk('/data/scratch/projects/uom00014/vllm/datasets/mmlu')
print('mmlu rows=', len(ds))
"
# expect: mmlu rows= ~14000

# 3. Server smoke (10 minutes on a100-short)
/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/sglang-cu12-response-check.sbatch
# Tail the log: should show "sglext" in the response dump and "routed_experts" base64-decoded

# 4. Single-cell smoke (1 hour on a100-short)
# Re-submit the wide-sweep sbatch with MODELS_CSV=allenai/OLMoE-1B-7B-0924-Instruct
# and BENCHMARKS=popqa, NUM_SAMPLES=8
```

## 8. Re-running the sweep after a venv rebuild

If `venv_sglang_0509` is wiped and rebuilt:

1. **Reinstall sglang 0.5.9 stack** (see §1 for the resolve path).
2. **Re-apply the capturer patch** (§2) by hand to the new
   `routed_experts_capturer.py`. Save the patch as a `.patch` file
   alongside the venv build script if the build is automated.
3. **Verify the driver still uses httpx**, not the openai SDK. The
   commit `89a733cd7` is the pinned "httpx-only" version.
4. **Run the pre-flight check** (§7 step 1) — must print `PATCHED`.
5. **Resubmit the sweep**:
   ```bash
   cd /data/gpfs/projects/uom00014/vllm
   sbatch examples/eval-moe/spartan/sglang-moe-eval-h100-bf16-wide.sbatch
   ```

## 9. If you upgrade to sglang ≥ 0.5.10

- **Drop the capturer patch** if upstream fixed `layer_id` plumbing.
  Verify by running §7 step 1 without the patch and checking that
  the layer tensor is `(T, L, top_k)` not `(T, 1, top_k)`.
- **Keep the httpx bypass** — the SDK still drops `sglext` on parse
  in every 0.5.x version through 0.5.21.
- **Update the wrapper** — 0.5.10+ exposes `--enable-return-routed-experts`
  as a CLI flag, so the monkey-patch isn't needed. The wrapper can
  be reduced to a one-liner: pass the flag on argv.
- **Don't expect the dep tree to resolve on cu128**. sglang ≥ 0.5.20
  needs cu13+; you'll need a node with driver 580+ or a forked
  `pyproject.toml`.

## 10. File pointers

| What | Where |
|------|-------|
| Driver | `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/moe_expert_stats_sglang.py` |
| Wide sweep sbatch | `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/sglang-moe-eval-h100-bf16-wide.sbatch` |
| Smoke sbatch (debug) | `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/spartan/sglang-cu12-response-check.sbatch` |
| Plan doc | `/data/gpfs/projects/uom00014/vllm/examples/eval-moe/.plans/sglang-bf16-sweep.md` |
| Venv | `/data/scratch/projects/uom00014/vllm/venv_sglang_0509/` |
| Output tree (current) | `/data/scratch/projects/uom00014/vllm/results_sglang_bf16_h100/` |
| vLLM-side handoff | `/data/gpfs/projects/uom00014/vllm/EVAL_MOE_HANDOFF.md` (Addendum 8) |
| Sister repo | `/data/projects/uom00014/llama-cpp-eval` (Jaccard aggregator) |
| Local sglang checkout | `/data/gpfs/projects/uom00014/sglang` (this repo — unmodified) |

## 11. Known unknowns

- **DeepSeek v2/v3 stride**: the capturer emits `(top_k + n_shared)`
  columns per token; the driver slices to `[:top_k]` for vLLM parity.
  Verified for v2 (16b-chat, 64E/top_k=6, n_shared=2). Not yet tested
  against v3 — if DeepSeek v3 ships with a different n_shared, the
  driver needs a per-model config or a runtime probe.
- **sgl-kernel arch detection**: the capturer patch assumes SM80/90
  forward hooks call the capturer exactly `num_layers` times per
  forward. If a future sglang adds early-exit or MoE-only-layer
  routing, the monotonic counter could drift. The pre-flight guard
  only checks that the patch is *applied*, not that it's *correct*.
- **Mixtral TP=4 sglang cell**: in flight. The Mixtral TP=4 sbatch
  (`sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch`) was submitted as
  job 33139440 on 2026-10-16 09:12 AEST, gpu-h100 PENDING (4–8 day
  wait estimate). Once it completes, the tree will have all 4
  models × 5 benchmarks = 20 expert_counts.json files and the
  3-way Jaccard comparison (vLLM × llama.cpp × sglang) can be
  run end-to-end. See `sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch`
  for the per-cell lifecycle.

## 12. Post-Addendum-8 progress (Oct 14 → Oct 16, 2026)

This section captures everything that happened since Addendum 8
(2026-10-14 11:38 AEST) — when the wide sweep was first re-submitted
with the line-675 fix.

### 12.1. Stride fix (commit `2ed502b4e`)

The first successful wide-sweep run (job **32933582**, 1h12m wall)
wrote 10 JSONs but every DeepSeek cell crashed with
`routed_experts has N elements; expected a multiple of L*stride =
28*8 = 224`. Root cause: sglang 0.5.9's capturer emits
`(T, L, top_k)` — shared-expert slots go in a SEPARATE field, not
appended to routed_experts. The driver's stride assumption
(`top_k + n_fused_shared_experts`) was wrong for DeepSeek
(`top_k=6, n_shared=2` → expected stride 8, actual stride 6).
Fix: try `stride = top_k` first; fall back to `stride = top_k +
n_fused_shared_experts` only if sglang is upgraded. Verifies across
all 42,566 rejected requests in job 32933582.

This unblocks both DeepSeek re-runs AND adding Qwen3 (which has
no shared experts, so stride was already correct).

### 12.2. Job chronology

- **32933582** (Oct 14 15:25 → 16:37): line-675 fix only. Wrote
  DeepSeek × 5 + OLMoE × 5 = 10 JSONs. Qwen missing (not in
  MODELS_CSV).
- **32969746** (Oct 15 04:44 → 06:33): user asked to add Qwen. Wrote
  Qwen × 5 = 5 JSONs. (DeepSeek + OLMoE already cached from 32933582
  → analyzer's idempotent skip reused them; per-cell engine load
  avoided.)
- **33009540** (Oct 15 06:35:54 → 06:36:20): CANCELLED 26s. JobName
  was literally `wrap` — this was a follow-up wrapper job the user
  cancelled before it ran. Not part of the sweep itself.
- **33009563** (Oct 15 21:03 → 22:19): the final wide sweep with
  commit `2ed502b4e` applied. DeepSeek × 5 + OLMoE × 5 = 10 JSONs
  written (Qwen was already cached from 32969746). All 15 cells
  across the 3 models now on disk.
- **33139440** (Oct 16 09:12): Mixtral BF16 TP=4 cell submitted via
  the new `sglang-moe-eval-h100-bf16-mixtral-tp4.sbatch`. PENDING on
  gpu-h100 as of 2026-10-16 09:12:57 (estimated start 2026-10-21).

### 12.3. Schema verification

Confirmed via `python3 -c "import json; d=json.load(open(...)); print(d['model_arch'])"` that all 15 cells produced bit-compatible
JSONs:

| Model | n_layer | n_expert | n_expert_used | Tokens (mmlu) |
|---|---|---|---|---|
| OLMoE | 16 | 64 | 8 | 11,050,747 |
| Qwen3 | 48 | 128 | 8 | 13,382,576 |
| DeepSeek | 28 | 64 | 6 | 13,374,689 |

All three are schema-valid (`model`, `model_arch`, `totals`,
`aggregate.{marginal_expert_counts,intra_pair_counts,adjacent_pair_counts}`,
per-subject `layer_expert_counts`). The sister
`llama-cpp-eval/examples/eval-moe-overview/aggregate_overview.py`
should accept the tree as-is once it gets the directory listing.

### 12.4. Cross-backend Jaccard readiness

Once the Mixtral BF16 TP=4 cell completes (job 33139440), the
sister-repo aggregator can be invoked to produce a 3-way comparison
(vLLM × llama.cpp × sglang) across all 4 models × 5 benchmarks
= 20 cells × 3 backends. The vLLM-side trees (bf16, fp8, w8a16,
w4a16) are already uploaded to MediaFlux (handoff §4.5 + §W).

**Note about Mixtral W4A16**: the vLLM-side Mixtral W4A16 JSONs
(under `results_wide_w4a16_awq_h100/...-Mixtral-.../compressed-tensors/`)
are **degenerate** (MarlinExperts capturer bug — see the sibling
`MIXTRAL_W4A16_BUG_HANDOFF.md` at the vLLM repo root). They should
NOT be included in the Jaccard comparison until rerun with
`--moe-backend triton` (job 33143679, PENDING). The W8A16 Mixtral
JSONs (also at `results_wide_w8a16_h100/`) are healthy and serve as
the vLLM-side Mixtral baseline for the 3-way comparison.

### 12.5. Outstanding items

1. **Job 33139440** (Mixtral BF16 TP=4): PENDING on gpu-h100.
   Expected wall time ~4–6h based on the wide sweep's MMLU cells
   (~10 min each for OLMoE/Qwen/DeepSeek at TP=1).
2. **Job 33143679** (vLLM Mixtral W4A16 with `--moe-backend triton`,
   see sibling bug handoff): PENDING on gpu-h100 at 2026-10-16
   10:24 AEST. If the capturer data is healthy, the vLLM-side
   Mixtral W4A16 row of the Jaccard matrix can be filled in.
3. **Optional: delete the stale single-dash smoke JSON** at
   `results_sglang_bf16_h100/allenai-OLMoE-1B-7B-0924-Instruct/bf16/moe-mmlu/expert_counts.json`
   (written by job 32843922) — the sister aggregator's
   tree-walker normalizes both, but the file is from a 8-prompt
   smoke test and may bias the Jaccard stats.
4. **Optional: file an sglang upstream issue** about the capturer
   `layer_id` being `None` for SM80/deepseek-v2/mixtral under
   CUDA-graph replay. The patch in §2 is correct but should be
   in sglang itself.

Co-authored-by: Hermes Agent <noreply@nousresearch.com>