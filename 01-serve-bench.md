# 01 — Handoff: `serve-bench` (serving-metrics measurement harness)

> Attach `00-shared-context.md` alongside this file. Read it first.

You are planning **`serve-bench`**, the measurement harness that underpins every cost/throughput/accuracy claim in the rest of the portfolio. Budget: **30 hours, July 2026.** It ships first and everything else cites it — if the measurement methodology isn't trustworthy, no downstream efficiency number is. Be skeptical and direct.

## Why this exists (and what it is NOT)
The portfolio's entire thesis is "I make models cheaper to serve," which is a *measurement* claim. serve-bench is the instrument. It is a benchmarking/measurement library — NOT a systems-trophy. An earlier dual-track version of this plan included lock-free queues and allocators here; those are CUT. No lock-free anything. If you find yourself designing concurrency primitives, stop — that's the wrong layer.

## Frozen spec (plan execution; don't redesign)
Records, per run:
- **Throughput:** tokens/sec at batch 1 and at batched settings.
- **Latency:** TTFT (time-to-first-token) and TPOT/ITL (per-output-token / inter-token latency), reported as P50/P90/P99 — never just mean.
- **Memory:** peak GPU memory.
- **Derived cost:** cost-per-million-tokens, given a disclosed GPU $/hr.
- **Accuracy delta:** a harness that runs perplexity + at least one task metric, so quantization's quality cost is always reported next to its speed gain.

Methodology discipline (non-negotiable): warmup before timing; coordinated-omission-safe load generation (schedule intended send times, don't just loop); full hardware/config/software-version disclosure; one disclosed benchmark host; every number regenerable by a single script.

## Resolved planning decisions (this session)

**Accuracy-delta harness — use a library, don't hand-roll.** The perplexity + task-metric harness is built on `lm-evaluation-harness`, not a bespoke perplexity implementation. Perplexity is easy to get subtly wrong (stride / sliding-window, tokenization), and using the standard tool is *more* credible than a custom one — provided the tool + version + exact task configs are disclosed. This counts as a disclosed library call, not hand-written work.

**Metric measurement is tiered (fixed-batch vs. under-load).** These measure two different things and must not be conflated:
- *Fixed batch-size* isolates the **engine's raw compute efficiency** (a property of the engine): batch-1 = latency floor, large batch = throughput ceiling, the sweep between = the efficiency curve. Low-variance, needs no CO handling.
- *Under-load (open-loop, arrival-rate driven)* measures **whole-system serving behavior** (queuing, scheduling, goodput under an SLO). This is the number that maps to real cost-per-token, but it requires CO-safe generation and a server loop the engine may not have.

Decision for the 30h budget:
- **Tier A (must — the backbone, never sacrificed):** fixed batch-size sweep for every engine/config. Batch-1 latency floor + a sweep to the throughput ceiling, with TTFT/TPOT reported as P50/P90/P99 at each point. This is what nano-infer can be compared on apples-to-apples.
- **Tier B (build the hook, report one number — first descope candidate):** a minimal CO-safe open-loop load generator (scheduled Poisson arrivals) producing a single headline: **goodput = max sustained QPS under a stated P99 SLO.** Run only against engines with a real server loop (vLLM, llama.cpp server). nano-infer gets a Tier-B number only if/when it grows a batching server; otherwise state plainly it is benchmarked at fixed batch, and why.

**Cost-per-token is reported both ways, with disclosure.** Always report cost-per-million-tokens at *max throughput* AND *at the stated SLO* (from Tier B where available). Never report a lone max-throughput cost number — that is the metric everyone games.

**Apples-to-apples pinning (the real credibility risk).** Any cross-engine comparison must pin and disclose, identically across engines: precision (weights + activations + KV), KV-cache config, sampling params (greedy vs. temperature/top-p), the exact prompt set and output-length policy, and the tokenizer. A comparison that doesn't pin these is where benchmarks quietly lie; the harness records all of them per run.

**Compute host — rent, don't buy (yet).** No usable GPU is owned; dev is on Windows 11; a RunPod instance is available. serve-bench runs on RunPod: cheaper in expectation than a ~$700 used 3090 + host for a one-year project, Linux-native (avoids the Windows/vLLM tax), and a pinned SKU + Docker image makes "one disclosed benchmark host" a reproducibility asset (anyone can rent the identical SKU and re-run the image). Freeze ONE GPU SKU as the disclosed host; use dedicated/secure-cloud for final headline runs, community pods for dev; lock clocks (`nvidia-smi -lgc`) where possible, disclose if not. Revisit buying at the nano-infer boundary (heavy interactive CUDA dev), decided with real GPU-hrs/week from serve-bench.

**Language — Python harness; the rigor lever is not the language.** Python orchestration around any engine (the whole ecosystem — vLLM benchmarks, lm-evaluation-harness — is Python). Timing is captured at the **CUDA-event boundary** (separate prefill/TTFT from decode/TPOT), never Python wall-clock, which leaks tokenizer/serialization overhead. The only defensible non-Python component is the Tier-B open-loop load generator's send-scheduling loop (Rust) — built in Python first and ported only if measurement shows Python GIL/GC jitter is a material fraction of inter-arrival times (unlikely at 1B / single 24GB / low QPS). Off-the-shelf `tokio` only; no hand-authored concurrency primitives (systems-trophy work stays cut).

## Dig-deeper questions for this session
1. **Architecture.** What's the cleanest design for a harness that wraps both nano-infer (a from-scratch engine) AND external baselines (llama.cpp, vLLM) behind one measurement interface, so comparisons are apples-to-apples? Language/structure recommendation (Python orchestration around any engine? a thin C++ timing core?) with reasoning.
2. **TTFT/TPOT measurement correctness.** How to measure these without contaminating them — streaming-token timestamping, separating prefill from decode, avoiding tokenizer/serialization overhead leaking into the numbers. What are the common ways people measure these wrong?
3. **Cost-per-token derivation.** The right way to turn throughput + GPU $/hr into a defensible cost-per-million-tokens figure (utilization assumptions, batch-size dependence, whether to report at fixed SLO vs. max throughput). What assumptions must be disclosed for the number to be credible?
4. **Accuracy-delta harness.** Which perplexity dataset(s) and which task metric give a credible, cheap, reproducible quality signal for a 1B model under quantization? How to keep this fast enough to run every config without becoming its own project.
5. **Coordinated omission for serving.** What does CO-safe load generation look like specifically for LLM serving (open-loop request arrival vs. closed-loop), and when does it actually matter for these metrics?
6. **Statistical rigor.** Run counts, warmup discipline, how to report variance, how to detect a noisy benchmark host, thermal/clock-throttling pitfalls on a consumer GPU.

## Open questions to ask the person first
- Exact GPU model and CUDA/driver versions (the benchmark host).
- Will baselines (llama.cpp, vLLM) run on the same card, same precision settings?
- Preferred orchestration language (Python is the likely answer given the ML stack).

## Produce
A 30-hour milestone plan, the harness API/architecture sketch, the methodology checklist that downstream artifacts will inherit, the accuracy-delta dataset/metric choice with justification, a risk register, and a definition-of-done. The methodology blog post ("Benchmarking serving cost without lying to yourself," August) is a downstream deliverable — outline what it must contain so this harness backs every claim in it.
