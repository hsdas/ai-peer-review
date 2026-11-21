# AI Peer Review (local-models edition)

This fork of [`poldrack/ai-peer-review`](https://github.com/poldrack/ai-peer-review)
generates AI peer reviews of academic papers using **locally-served LLMs only**.
The original used cloud LLM APIs (OpenAI, Anthropic, Google, Together.ai); this
fork removes every cloud API call so the manuscript text never leaves a machine
you control. This is the intended mode for peer review of **unpublished**
manuscripts you intend to submit to a Q1 SCI-indexed journal.

> **What this is and is not.** This tool is a **pre-submission self-check**:
> it surfaces scientific concerns before you submit, the way a critical
> co-author or a friendly reviewer would. It is **not** a substitute for the
> journal's actual peer-review process, and a positive AI review does not
> imply acceptance. Treat its output as one signal among several.

## Why local-only

When you send an unpublished manuscript to a third-party LLM API, the provider
may log, retain, or use the prompt content for model training, and the data
transits networks you do not control. For unpublished work heading to a Q1 SCI
journal, that risk is unacceptable. This fork makes the tool **structurally
incapable** of contacting a third-party LLM provider: the cloud client files
(`openai_client.py`, `anthropic_client.py`, `google_client.py`,
`deepseek_client.py`, `llama_client.py`) and their API-key plumbing have been
deleted. The only LLM client that remains, `LocalLLMClient`, talks to an
OpenAI-compatible inference server **you** run.

There is no `OPENAI_API_KEY`, no `ANTHROPIC_API_KEY`, and no network call to
`api.openai.com` / `api.anthropic.com` / `generativelanguage.googleapis.com` /
`api.together.xyz`. Even if you tried to re-enable a cloud model, there is no
code path left that would send the manuscript to one.

## Features

- Submit a paper (PDF) for review by one or more local LLMs
- Generate an independent peer review per reviewer
- Generate a meta-review that aggregates the individual reviews (reviewer
  identifiers are stripped first, so the meta-reviewer cannot tell which
  review came from which model)
- Generate a concerns table (CSV) marking which reviewer raised each concern
- Store everything in markdown, CSV, and JSON
- Ships a **scientific-validity-only** review profile that suppresses
  presentation nitpicks and focuses on methods, inference, statistics,
  reproducibility, external validity, and claim-to-source support

## Supported local servers

Any server that implements the OpenAI `/v1/chat/completions` endpoint works.
Common choices (default base URL shown):

| Server | Default base URL | Notes |
| --- | --- | --- |
| [Ollama](https://ollama.com) | `http://localhost:11434/v1` | Easiest; one-command model pull. Default for this fork. |
| [LM Studio](https://lmstudio.ai) | `http://localhost:1234/v1` | GUI; good for trying models before committing. |
| [llama.cpp `llama-server`](https://github.com/ggml-org/llama.cpp) | `http://localhost:8080/v1` | Minimal; supports GGUF quants. |
| [vLLM](https://github.com/vllm-project/vllm) | `http://localhost:8000/v1` | Highest throughput; best for serving several models at once on a multi-GPU box. |
| [LocalAI](https://localai.io) | `http://localhost:8080/v1` | Drop-in for OpenAI clients. |
| [TGI](https://github.com/huggingface/text-generation-inference) (with `--chat-template`) | `http://localhost:8080/v1` | Hugging Face's serving stack. |

## Recommended models for Q1 SCI peer review

For peer review at Q1 SCI quality you want **strong reasoning** and a **long
context window** (a full manuscript plus references is often 20-50K tokens,
and the meta-review input is every individual review concatenated, easily
30-60K tokens). The models below are open-weight and current as of late 2026;
verify exact Ollama tags with `ollama list` / `ollama show <model>` (tags
change).

| Model | License | Context | Why it fits | Rough VRAM (4-bit) |
| --- | --- | --- | --- | --- |
| **Kimi K3** (Moonshot) | open weights | 1 M | Top open model on Terminal-Bench 2.1 (Morph, Nov 2026); 2.8 T total / 104 B active MoE | large multi-GPU server |
| **DeepSeek-R1** / **DeepSeek V3.2** | MIT | 64 K – 128 K | Explicit reasoning chain; strong on methods/statistics critique | 70 B distill ~40 GB; full V3 ~190 GB |
| **Qwen3-235B-A22B** | Apache 2.0 | 256 K | Top-tier general + scientific reasoning; MoE so inference is cheaper than the parameter count suggests | ~130 GB |
| **Qwen3-32B** | Apache 2.0 | 128 K | Strong balance; the shipped default. Fits one 24 GB GPU | ~20 GB |
| **Llama 4 Scout** | Llama 4 license | 10 M | Best-in-class long context if you feed very large manuscripts + supplements | large multi-GPU |
| **Llama 3.3 70B Instruct** | Llama 3.3 license | 128 K | Solid, well-understood general reasoning | ~40 GB (2×24 GB) |
| **GLM-5.1** (Zhipu) | open weights | 128 K | Reported SOTA on community leaderboards (r/LocalLLaMA, Apr 2026) | model-size dependent |
| **Gemma 3 27B IT** | Gemma terms | 128 K | Smaller but punchy; good "second opinion" reviewer | ~16 GB |

Sources: Morph "Best Open Source LLMs (Nov 2026)" (`morph.so`); r/LocalLLaMA
"Best Local LLMs - Apr 2026"; Hugging Face open-weights timeline
(`hidekazu-konishi.com`, Jul 2026); DeepSeek model guide (Apr 2026).

**Practical guidance.** A 27-32 B model on a single 24 GB consumer GPU (RTX
3090/4090/A5000) is the realistic entry point and already meaningfully better
than the 7-13 B models that are too weak for serious peer review. For Q1 SCI
rigor, prefer ≥ 70 B (Llama 3.3 70B, DeepSeek-R1 70B distill) or a strong
reasoning model (DeepSeek-R1, Qwen3-235B). If you do not have local GPU
hardware, you can rent a GPU instance you fully control (e.g. RunPod, Lambda
Labs, Vast.ai) and point this tool at the server you start there — note this is
a pragmatic middle ground: you control the server and no third-party LLM API
sees the prompt, but the manuscript does physically leave your premises, so
pick a provider and region you trust and shut the instance down when done.

## Installation

```bash
git clone https://github.com/hsdas/ai-peer-review.git
cd ai-peer-review

# with uv (recommended; regenerates the lockfile for the trimmed deps)
uv sync

# or with plain pip
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

Requires Python ≥ 3.10. Dependencies are now minimal: `click`, `pypdf`,
`openai` (used as the OpenAI-compatible client for the local server),
`python-dotenv`, `pandas`, and `pytest`. The cloud SDKs (`anthropic`,
`google-generativeai`) and unused helpers (`tiktoken`, `tqdm`, `structlog`,
`pydantic`, `ipykernel`, `requests`) have been removed.

## Quick start (single reviewer)

1. Start a local server and load a model. With Ollama:

   ```bash
   ollama pull qwen3:32b
   ollama serve   # already running if `ollama pull` worked
   ```

2. Point the tool at it and run a review:

   ```bash
   # set the default local server once
   ai-peer-review config server --base-url http://localhost:11434/v1

   # the shipped default reviewer is named "reviewer-1" and uses qwen3:32b;
   # change it if you loaded a different model:
   ai-peer-review config add reviewer-1 --model qwen3:32b
   ai-peer-review config meta     --model qwen3:32b

   # review a paper
   ai-peer-review review path/to/paper.pdf
   ```

Outputs land in `./papers/<paper-name>/`:
`review_reviewer-1.md`, `meta_review.md`, `concerns_table.csv`, `results.json`.

## Multi-reviewer setup (genuine multi-perspective)

The original tool's strength is running **several distinct models** so the
meta-reviewer aggregates genuinely independent perspectives, the way multiple
human reviewers would. To do this locally, add one entry per model and either
serve them on separate ports or review against one model at a time:

```bash
# Three reviewers, each a different model family
ai-peer-review config add reviewer-qwen     --model qwen3:32b
ai-peer-review config add reviewer-deepseek --model deepseek-r1:70b
ai-peer-review config add reviewer-llama    --model llama3.3:70b

# Use a strong/long-context model for the meta-review
ai-peer-review config meta --model qwen3:32b

ai-peer-review list-models
ai-peer-review review path/to/paper.pdf --models reviewer-qwen,reviewer-deepseek,reviewer-llama
```

If your hardware cannot hold several large models at once, run the same
command multiple times with `--models reviewer-qwen`, then
`--models reviewer-deepseek`, etc. — existing reviews are reused unless you
pass `--overwrite`, so each run only calls the model you name. When all
individual reviews exist, run once more (with any `--models` subset, or all of
them) to generate the meta-review over the full set.

## Configuration

Configuration lives in `~/.ai-peer-review/config.json` (or the path you pass
to `--config-file`, or `$AI_PEER_REVIEW_CONFIG_FILE`). It has four sections:

```jsonc
{
  "local_server": {                 // inherited by every reviewer unless it overrides
    "base_url": "http://localhost:11434/v1",
    "api_key": "local"              // ignored by local servers; SDK requires non-empty
  },
  "local_models": {                 // one entry per reviewer
    "reviewer-1": {
      "model": "qwen3:32b",         // tag as recognized by your local server
      "base_url": null,             // null => inherit local_server.base_url
      "api_key": null,
      "temperature": 0.1,
      "max_tokens": 8000
    }
  },
  "meta_reviewer": {                // the model used for meta-review + concerns JSON
    "model": "qwen3:32b",
    "base_url": null,
    "api_key": null,
    "temperature": 0.1,
    "max_tokens": 8000
  },
  "prompts": { "system": "...", "review": "...", "metareview": "...", "concerns_extraction": "..." }
}
```

### CLI config commands

```bash
ai-peer-review config show                                 # print local-model config
ai-peer-review config server --base-url URL [--api-key K]  # set default server
ai-peer-review config add NAME --model TAG [--base-url URL] [--temperature T] [--max-tokens N]
ai-peer-review config remove NAME                          # remove a reviewer
ai-peer-review config meta --model TAG [--base-url URL] [--temperature T] [--max-tokens N]
ai-peer-review list-models                                 # list configured reviewers
```

### Environment variables (quick single-model override)

| Variable | Purpose |
| --- | --- |
| `AI_PEER_REVIEW_BASE_URL` | Base URL of the local server (overrides config) |
| `AI_PEER_REVIEW_API_KEY` | Placeholder API key (overrides config) |
| `AI_PEER_REVIEW_CONFIG_FILE` | Path to a custom config JSON |

Copy `.env.example` to `.env` to set these without exporting them each shell.

## Scientific-validity-only profile

`configs/scientific-only.json` ships a ready-to-use prompt profile that
instructs the models to raise a criticism **only when resolving it could change
the validity, support, scope, reproducibility, or interpretation of a
scientific claim**, and to exclude concerns whose only consequence is
formatting, length, readability, layout, grammar, or prose style. This is the
recommended profile for Q1 SCI submission prep — it focuses the review on
methods, inference, statistics, reproducibility, external validity,
demonstrated novelty of explicit novelty claims, and claim-to-source support,
and asks for a recommendation that would be unchanged if the paper were
reformatted without changing its evidence or claims. The profile now also
carries a `local_models` / `meta_reviewer` block (two reviewers: Qwen3 + a
DeepSeek-R1 reasoning reviewer) so it works out of the box locally.

```bash
ai-peer-review --config-file configs/scientific-only.json review path/to/paper.pdf
```

The profile keeps the original placeholders (`{paper_text}`, `{reviews_text}`,
`{meta_review_text}`, `{model_names}`, `{first_model}`, `{second_model}`,
`{model_mapping}`), the `CONCERNS_TABLE_DATA` marker, the fenced JSON block,
and the NATO phonetic reviewer names, so the interpolation and output-parsing
contracts the pipeline relies on are preserved.

### CPU-only / low-RAM machines (no GPU, ~8 GB RAM)

`configs/cpu-small.json` is the same scientific-validity profile retuned for a
modest CPU-only box (e.g. an Intel i5-3450 with 8 GB DDR3 and no GPU). It
configures a single **Phi-4-mini** reviewer (3.8 B, ~2.5 GB in 4-bit, 128 K
context — the strongest model that fits alongside the OS and this tool in
8 GB) and lowers `max_tokens` to 4000 to bound generation time on slow CPU
inference. Expect a full review to take 10-30 minutes per paper on such
hardware, and treat the output as a first-pass screening check rather than a
rigorous review: a 3.8 B model catches obvious issues (sample-size
over-generalization, missing uncertainty, unsupported causal claims) but
misses subtler methodological flaws a 70 B+ model would surface.

```bash
ai-peer-review --config-file configs/cpu-small.json review path/to/paper.pdf
```

**Older CPUs without AVX2** (Ivy Bridge and earlier, ~2012): Ollama bundles an
AVX-only runner and should auto-select it, but if `ollama serve` fails to
start or crashes on model load, fall back to `llama.cpp`'s `llama-server`
(which compiles for the host CPU's instruction set, including AVX1-only) and
point the tool at it — the `LocalLLMClient` works with any OpenAI-compatible
server, not just Ollama. See the troubleshooting section.

## Outputs

For each paper (default `./papers/<paper-name>/`):

- `review_<reviewer-name>.md` — individual review from each local reviewer
- `meta_review.md` — synthesized meta-review (reviewer identities stripped)
- `concerns_table.csv` — one row per concern, one boolean column per reviewer
- `results.json` — `individual_reviews`, `meta_review`, `reviewer_to_model`

## Applying these changes to your fork

This directory is the modified fork. Two ways to get it onto your GitHub fork:

1. **Copy in place.** Copy these files over your working copy, commit, push:

   ```bash
   cd /path/to/your/ai-peer-review
   cp -r /home/z/my-project/download/ai-peer-review-local/. .
   git add -A
   git commit -m "Replace cloud LLM APIs with local OpenAI-compatible models"
   git push
   ```

2. **Apply as a patch.** A `local-models.patch` is shipped alongside this
   directory; from the root of your fork:

   ```bash
   git apply --whitespace=fix local-models.patch
   ```

   (The patch is generated against the upstream commit your fork was created
   from; re-generate it with `diff -ruN <upstream> <this-dir>` if it no longer
   applies cleanly.)

See `CHANGES.md` for the full, file-by-file change summary.

## Troubleshooting

- **`Connection refused` / `openai.APIConnectionError`** — the local server is
  not running or the `base_url` is wrong. Check `curl http://localhost:11434/v1/models`
  (Ollama) responds, and that `ai-peer-review config show` shows the URL you
  expect.
- **`model not found`** — the model tag in the config does not match what the
  server has loaded. With Ollama, run `ollama list` and copy the exact tag.
- **Truncated reviews** — raise `max_tokens` for that reviewer:
  `ai-peer-review config add reviewer-1 --model qwen3:32b --max-tokens 12000`.
- **Slow generation on a large model** — pass a longer timeout by editing the
  reviewer entry, or use a smaller / distilled model. DeepSeek-R1 reasoning
  traces are long; expect minutes per review.
- **Concerns table is empty** — the meta-reviewer did not emit the
  `CONCERNS_TABLE_DATA` marker. Re-run with `--overwrite`, or use the
  scientific-validity-only profile, which is more reliable about the marker.
- **Out-of-memory on GPU** — drop to a smaller quant or a smaller model. With
  Ollama, the default tag is usually a 4-bit quant; a `:32b` model on a 24 GB
  GPU is near the limit — close other VRAM consumers.

## License

Inherits the upstream license (see `LICENSE`).
