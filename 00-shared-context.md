# 00 — Shared Context (read first, attached to every handoff)

You are helping plan/execute ONE piece of a single-track ML-infrastructure portfolio. This file is the shared background; a second file gives you the specific artifact and the questions to dig into. You have none of the originating conversation's context — everything you need is in these two files. Be skeptical, direct, and non-sycophantic; the person who commissioned this explicitly wants brutal honesty and evidence over encouragement, and has corrected prior sessions toward honesty multiple times.

## Who this is for
Joshua Labasbas — May 2026 BS Computer Science, University of Florida (GPA ~3.3–3.4, not on resume). Now a full-time Firmware Validation Engineer at Tesla (Palo Alto, converted from a 2025 internship). Real, verifiable strengths: led a CT-to-MRI Brownian Bridge Diffusion Model (SSIM 82.5%, distributed training on 8×B200 GPUs on an HPC cluster, diagnosed and recovered a regression); built containerized co-simulation infra at Tesla (Docker, Rust IPC, simulated CAN, 144× test-cadence improvement, 99.7% cost reduction); production avionics firmware at Dynon; a custom C++23 game engine with deterministic fixed-timestep simulation; RoboSub AUV work (PID, YOLOv7). Skill line: Python, C/C++, SQL, Rust, GLSL; PyTorch, NCCL, FlashAttention, vLLM (USAGE level — no contributions yet), OpenGL. **Gaps to respect honestly:** no merged OSS PRs anywhere, no CUDA kernel authorship yet, no publications on resume (a MIDL 2026 submission is in progress off-resume). CUDA experience is usage-level (via the diffusion project), not authorship.

## The single thesis the whole portfolio serves
**"I make models cheaper to serve."** The portfolio targets the inference-cost-optimization layer of ML infra — quantization, speculative decoding, KV-cache optimization, serving throughput — chosen because deep research found it the most durable and *counter-cyclical* sub-layer: when AI capex decelerates (consensus forecasts a 2027–2028 deceleration) and the revenue-to-capex gap dominates, cutting cost-per-token becomes MORE valuable, not less. Inference passed ~half of AI compute in 2025, heading toward two-thirds in 2026. Every artifact must move a cost / throughput / accuracy / memory number, or prove a capability that does.

HFT was deliberately CUT from an earlier dual-track version of this plan; do not reintroduce trading/order-book/lock-free-as-artifact work. The NYC option is preserved anyway because GPU-performance roles at trading firms (HRT, XTX, Jane Street, Citadel Securities) hire on exactly this efficiency signal for their own research clusters.

## The whole portfolio (so you know where your piece fits)
Single-track, ~595 hours, ~13.5 hrs/week, July 2026 → June 2027, apply April–June 2027. Six Tier-0 pieces:
1. **serve-bench** (30h) — serving-metrics measurement harness (cost-per-token, throughput, TTFT/TPOT, memory, accuracy-delta). Credibility layer; ships first.
2. **nano-infer** (230h) — from-scratch CUDA/C++ inference engine for Llama-3.2-1B; headline is an efficiency cost-frontier via quantization + speculative decoding. The flagship.
3. **OSS ladder** (130h) — 2–3 merged vLLM/SGLang PRs in efficiency corners. The actual ML-infra hiring currency; escalated.
4. **GPU MODE** (60h) — 2–3 efficiency-relevant kernel problems + optimization-journey writeups; feeds nano-infer.
5. **Paper trail** (25h) — methodology post (Aug), cost-reduction post (March, Show-HN), optional production-stack case study.
6. **Interview prep** (120h) — single ML-infra track, deep; inference economics fluency.

## Hardware & money
Used 24GB CUDA card (3090-class, ~$600–900), bought week one. H100 rental for final headline benchmarks only (~10–20 hrs ≈ $30–60 at ~$1.50–3/hr specialist clouds). Total cash ~$650–1,000. Money is not the constraint; hours are.

## Shared methodology rules (apply to every measured artifact)
- Every number regenerable by one script. One disclosed benchmark host. Full hardware/config disclosure.
- Latency as percentiles (P50/P90/P99), never just averages. Warmup discipline. Coordinated-omission-safe load generation.
- Quantization speed gains ALWAYS reported alongside their accuracy cost (perplexity + a task metric).
- State limitations before a reader can raise them (single-GPU, 1B model, file-replay benchmarks, etc.).
- Be explicit about what is hand-written vs. a library call. Getting caught implying otherwise is fatal to credibility.

## Gates & sacrifice order (portfolio-level)
Key dates: Week 1 GPU owned · Aug 31 serve-bench + methodology post live · Sep 30 OSS rung-1 submitted · Dec 31 nano-infer quantization frontier working-or-descope · February mock + Feb 28 OSS fallback · Mar 15 Tier-1 gate · April–May applications.
Sacrifice order if crunch hits: GPU MODE → writeup-only; drop FP8 activations (keep INT4 weight-only + frontier); drop quantized KV; drop 2nd/3rd OSS rung; drop case-study post. NEVER sacrificed: methodology post, ≥1 merged OSS PR, the cost/accuracy frontier table, the durability-positioning prose, the prep surge.

## How to behave in your session
- Plan/execute the piece; do not redesign the portfolio or reintroduce HFT.
- Flag genuine blockers, but resolve spec conflicts against the thesis ("make models cheaper to serve"), the ledger, and the sacrifice order rather than rewriting the architecture.
- Ask the person the open questions listed in your artifact file before producing a final plan — the plan is only as good as those inputs (their exact GPU, their baselines, their current skill level on the specific topic).
- Produce a concrete deliverable: milestone plan with hour estimates that sum to the artifact's budget, decision lists, a risk register, and a definition-of-done checklist.
