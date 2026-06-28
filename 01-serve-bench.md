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
