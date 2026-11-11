# Research: a per-model correctness classifier for the Lonn router

Research for the `prototype3.0` router (2026-10-22). The question: how should the router predict **P(correct | prompt, model)** for each pool model, and then pick the cheapest model that is good enough? Claims cite primary sources (papers, official repositories, model cards). Lonn's own numbers come from `results/benchmark/` and `results/router_eval/`. Anything not verified is marked **(unverified)**.

## TL;DR

**Recommendation**

1. **Backbone:** start with a frozen, open embedder. `gte-modernbert-base` (149M parameters, Apache-2.0, 8k context) or `bge-base-en-v1.5` (MIT, 512 tokens) are both fine. Move to a fine-tuned **ModernBERT-base** (Apache-2.0) or **Ettin-150M** (MIT) encoder only after more labels exist. Two independent re-evaluations found that the choice of embedding backbone barely changes routing quality ([LLMRouterBench](https://arxiv.org/abs/2601.07206); [Li 2025](https://arxiv.org/abs/2505.12601)). Do not spend the project's effort there.
2. **Router method:** use a plug-in router in the style of [CARROT](https://arxiv.org/abs/2502.03261). Train a **multi-label head with one calibrated logistic output per pool model**, giving P̂(correct | x, m). Pair it with a **per-model expected-cost estimate**, meaning measured output-token volume × price, not list price alone. Choose the model by minimising expected cost subject to P̂ ≥ τ, or by maximising P̂ − λ·cost.
   - The baselines to beat are kNN (k ≈ 20–100) and linear/logistic regression on the same embeddings. Across four benchmark families these match or beat MLPs, graph routers and attention routers ([Li 2025, Table 2](https://arxiv.org/abs/2505.12601)).
   - Add an **IRT-style factorisation** (model ability per latent dimension × prompt difficulty and discrimination) as the second candidate. It is low-parameter, interpretable, and can absorb new models from a few answers ([IRT-Router](https://arxiv.org/abs/2506.01048); [tinyBenchmarks](https://arxiv.org/abs/2402.14992)).
3. **Labels:** the answer bank is the right **kind** of label, but it is far too small. Only **249 of the 690 questions** separate the models: 400 were answered correctly by all 8 models and 41 by none.
   - Grow the bank to roughly 5–10k questions **disjoint from the 690 evaluation questions**. At the measured rate of about $0.037 per question across all 8 models, that costs roughly $185–370 plus WritingBench judging.
   - Optionally pretrain the prompt side on public per-prompt correctness matrices: EmbedLLM, SPROUT, LLMRouterBench, RouterEval.
   - Use dataset-level accuracies only as a **prior or fallback**, never as per-prompt targets.
4. **Evaluation:** keep the existing replay harness, but report a **pass-rate vs cost curve** (a "deferral curve") over a sweep of λ or τ. Add these metrics:
   - area under that curve (AIQ / nAUC);
   - **QNC/CPT**: the cost needed to match Claude Sonnet 5.5;
   - **PGR vs the oracle**;
   - per-model calibration (Brier, ECE, AUROC);
   - a model-selection histogram (collapse check).

   Evaluate with k-fold or leave-one-dataset-out splits. Never score on prompts that were used for training.

**Verdict on Laya: do not keep it as the router's classifier.** It is an open, Apache-2.0 "decision engine" by Convai Innovations, built on a ModernBERT-large encoder with a 512-token window. Its interface answers natural-language questions ("which topic?", "is it hard?"), and neither answer predicts *which of our 8 models will get this prompt right*.
- **The measured result:** the true-topic ablation scores 78.6% and Laya scores 78.1–78.4%. Laya's complexity scores cluster at 0.25–0.62, and the router still sends 651/690 questions to one model.
- **What would be needed to keep it:** we would have to ask it 8 per-model yes/no questions, inside a 512-token window shared with the question text. We would also have to keep re-training its head with RLCD, and its question wording is coupled to the checkpoint. That gives up most of the benefit of a plain encoder with a multi-label head.
- **Recommendation:** use its backbone (ModernBERT) directly, and keep Laya only as an optional baseline.

## Where Lonn stands (from the repo)

| Policy (690 questions, 8 models, 8192 tokens) | Pass | Cost (USD) | Source |
|---|---:|---:|---|
| Router, stock Laya, α=0.5 | 78.1% | 1.10 | `results/router_eval/runs.csv` |
| Router, fine-tuned Laya | 78.4% | 1.15 | same |
| Router, true dataset topic + difficulty | 78.6% | 1.16 | same |
| Best fixed model (Claude Sonnet 5.5) | 88.1% | 7.58 | same |
| Per-prompt oracle (cheapest correct) | 94.1% | 1.08 | same |
| *Dataset oracle*: best model per dataset (in-sample) | 90.0% | 5.16 | computed from `results/benchmark/` |
| *Dataset oracle*: cheapest model within 2 pts of the best per dataset (in-sample) | 89.3% | 3.86 | computed |

Observations that drive the design:

- **The headroom has two parts.** A per-dataset prior, used well, already reaches about 89% at about $3.9 (optimistic, because it is chosen in-sample). The jump to 94% at $1.08 needs per-prompt prediction.
  - The current formula gets 78.6% even with true topics. That points to the policy, which uses static aptitudes and list-price normalisation, rather than to missing category information.
- **Signal is sparse.** In 400/690 questions every model is right, and in 41 none is. Only 249 questions carry information about *which* model to use. This is consistent with the "model-recall failure" that [LLMRouterBench](https://arxiv.org/abs/2601.07206) identifies as the main remaining gap: routers miss the few models that solve the hard items.
- **List price ≠ billed cost.** Average billed cost per question, from the bank:
  - gpt-oss-120b: $0.42 per 1k questions;
  - deepseek-v4.1-flash: $1.68 per 1k;
  - Claude Sonnet 5.5: $10.99 per 1k.

  gpt-oss-120b passes 75.8% at about a quarter of v4.1-flash's billed cost, yet the router prefers flash. Reasoning-token volume varies by model, so the router needs a **per-model cost estimate**. [CARROT](https://arxiv.org/abs/2502.03261) predicts cost per (prompt, model) for this reason.
- **Per-dataset accuracy varies sharply:**
  - WritingBench: 2% (gpt-oss-20b) to 53% (Muse Spark).
  - LiveCodeBench: 42% to 89%.
  - GPQA: 42% to 88%.

  These are the datasets where routing matters. MATH-500, HumanEval+ and ARC are saturated (≥ 90% for every model).

## Q1. Router architectures and training objectives

| Method | What it predicts / objective | Prompt encoder | Training data | vs kNN / linear baseline | Evidence |
|---|---|---|---|---|---|
| **RouterBench kNN / MLP** (Hu et al. 2024) | Per-model performance P_ij; route by λ·P − cost | all-MiniLM-L12-v2 (best); kNN k=40 | 70% of each task, 11 models | They *are* the baselines. They match the best single model at lower or similar cost, but "none … significantly outperform the baseline Zero router" | [arXiv 2403.12031 §5.1, App. B](https://arxiv.org/abs/2403.12031) |
| **RouteLLM** (Ong et al. 2024) | Binary win-prob strong vs weak. Variants: SW ranking (Elo-weighted similarity), matrix factorisation, BERT-base classifier, Llama-3-8B classifier | text-embedding-3-small (MF, SW); BERT-base / Llama-3-8B full fine-tune | 65k Arena pairs + GPT-4-judge or MMLU-gold augmentation | On Arena data alone, BERT and the causal LLM were ≈ random ("high capacity approaches performing worse in a low-data regime"). After augmentation, MF reached the best MT-Bench APGR of 0.802 vs random 0.500. On MMLU only +~20% over random. | [arXiv 2406.18665 Tables 1–3](https://arxiv.org/abs/2406.18665) |
| **Hybrid LLM** (Ding et al. 2024) | P[q(small) ≥ q(large) − t], a relaxed "easy" label; threshold tuned at test time | DeBERTa | Per-query response-quality samples (BART score) | Binary only; no kNN comparison | [arXiv 2404.14618 §3](https://arxiv.org/abs/2404.14618) |
| **RouterDC** (NeurIPS 2024) | Dual contrastive loss: query vs LLM embeddings (top-K⁺ / bottom-K⁻ models) plus sample–sample loss | mDeBERTaV3-base (86M) | Union of 5 task train splits, 7 × 7B models | +2.76 pts (ID) and +1.90 pts (OOD) average accuracy over the best routing baseline. No kNN baseline reported. | [arXiv 2409.19886 Tables 1–2](https://arxiv.org/abs/2409.19886) |
| **EmbedLLM** (ICLR 2025) | Matrix factorisation of the 0/1 correctness matrix (model embedding ⊙ projected question embedding → BCE) | all-mpnet-base-v2 | 112 models × 36,054 questions | Correctness accuracy 0.741 vs kNN 0.715 on the full data. At 1k questions 0.644 vs 0.637, so the gap is ~0.7 pts at small scale and ~2.6 pts at 29k. | [arXiv 2410.02223 Table 1](https://arxiv.org/abs/2410.02223) |
| **GraphRouter** (ICLR 2025) | Heterogeneous GNN over task/query/LLM nodes; edge prediction | BERT-type PLM; LLM nodes initialised from text descriptions | 4 task types | Claims ≥ 12.3% over its baselines (HybridLLM, FrugalGPT, C2MAB-V; no kNN). In [Li 2025](https://arxiv.org/abs/2505.12601) Graph (k=100) averages 51.82 AUC vs kNN 52.68. | [arXiv 2410.03834](https://arxiv.org/abs/2410.03834) |
| **IRT-Router** (ACL 2025) | MIRT: P = σ(aᵢᵀθ_m − bᵢ), with query discrimination aᵢ and difficulty bᵢ from the embedding and multi-dimensional model ability θ_m; plus NIRT with explicit ability dimensions; kNN "warm-up" of query vectors | bert-base-uncased; 25 dims | 20 LLMs × 12 datasets | Beats most baselines on reward. On a *new* LLM, ACC 0.67 / AUC 0.62. In [EquiRouter's](https://arxiv.org/abs/2602.03478) re-run, MIRT nAUC 0.748 vs kNN 0.754 on RouterBench. | [arXiv 2506.01048 §4, Table 4](https://arxiv.org/abs/2506.01048) |
| **CARROT** (2025) | Plug-in: multi-label P̂_m(correct \| x) **and** Ĉ_m(cost \| x), then argmin of a convex risk combination; minimax-rate optimal | text-embedding-3-small + kNN (k=40), or RoBERTa-base with a shared trunk and per-model final layer | SPROUT (≈44k prompts, 13–15 models), RouterBench, Open LLM LB v2 | Only "marginal improvements over the RouterBench router" (= cost-unaware kNN) on RouterBench. On SPROUT both beat the Zero router "substantially". "The large majority of routing performance is … correctly assessing the accuracy of each model." | [arXiv 2502.03261 §4](https://arxiv.org/abs/2502.03261) |
| **MixLLM** (NAACL 2025) | Per-LLM quality and length regressors (random forest / MLP / kNN) over tag-enhanced embeddings, plus a bandit meta-decision with a latency penalty | BERT-type, fine-tuned with InsTag tags | Mixed instruction data | 97.25% of GPT-4 quality at 24.18% of the cost (best baseline 96.39% at 32.94%) | [arXiv 2502.18482 §4.2](https://arxiv.org/abs/2502.18482) |
| **UniRoute** (Google, 2025) | Represents each LLM by its **per-cluster error vector on a small validation set**, so the router works for unseen LLMs | Gecko 1B | EmbedLLM, RouterBench, SPROUT, held-out LLMs | Statistically significant but small gain over kNN: EmbedLLM area 0.651 vs kNN 0.636 (clairvoyant MLP 0.664). On RouterBench, 0.711 vs 0.707. | [arXiv 2502.08773 App. F](https://arxiv.org/abs/2502.08773) |
| **Arch-Router** (Katanemo, 2025) | Generative 1.5B LM that maps a query to a *user-written policy* (domain/action), and the policy to a model | Qwen-2.5-1.5B-based (unverified: the paper names a 1.5B model; base family per model card) | Synthetic labelled conversations | Not a performance predictor: it encodes human preference, not P(correct). +7.71% over proprietary LLMs on policy matching. | [arXiv 2506.16655](https://arxiv.org/abs/2506.16655), [model](https://huggingface.co/katanemo/Arch-Router-1.5B) |
| **NVIDIA prompt-task-and-complexity-classifier** | 11 task types + 6 complexity dims → hand-weighted overall score | DeBERTa-v3-base, ~0.2B, 512 tokens | 4,024 human-annotated English prompts | A classifier of prompt *attributes*, not of model correctness. Its reported top-1 accuracy is 0.94–0.998 per head. NVIDIA Open Model License. | [model card](https://huggingface.co/nvidia/prompt-task-and-complexity-classifier) |
| **kNN-is-enough** (Li 2025, rev. 2026) | Utility prediction compared across kNN, linear, MF, MLP, graph and attention routers on identical BERT embeddings | BERT (SFR gives a "modest" gain without changing the ranking) | AlpacaEval, HELM-Lite, Open LLM, RouterBench | Average Pareto-AUC: **linear 53.14**, MLP-MF 52.88, kNN(100) 52.68, Graph 51.82, MLP 51.71, attention 47–50 (oracle 64.91, random 40.35). kNN degrades 2.63 pts out of distribution vs 6.67 for complex methods. | [arXiv 2505.12601 Tables 2–4](https://arxiv.org/abs/2505.12601) |
| **LLMRouterBench** (2026) | Unified re-evaluation of 10 routers, 33 models, 21 datasets, 400k instances | gte-qwen2-7B (swapping in MiniLM or nli-bert: "no significant difference") | Its own bank | "Many routing methods exhibit similar performance … several recent approaches, including commercial routers, fail to reliably outperform a simple baseline". The leaders sit close to the **Dataset Oracle** (best model per dataset). | [arXiv 2601.07206 §4.2](https://arxiv.org/abs/2601.07206) |
| **EquiRouter / routing collapse** (2026) | Pairwise logistic **ranking** loss over models (higher accuracy first, ties broken by cheaper), not scalar regression | Query trunk + model embeddings | RouterBench, MMR-Bench | nAUC 0.771 vs kNN 0.754. GPT-4-level quality at 77% of its cost vs 93.8% for kNN. Swapping the ranking loss for MSE raises QNC from 0.773 to 0.964. | [arXiv 2602.03478 Table 1](https://arxiv.org/abs/2602.03478) |
| **Avengers-Pro** (2025) | Embed + cluster queries; per-cluster performance–efficiency score; no neural training | Embedding model + k-means | 6 benchmarks, 8 frontier models | Matches the strongest model at 27% lower cost. In LLMRouterBench it is on par with EmbedLLM and GraphRouter. | [arXiv 2508.12631](https://arxiv.org/abs/2508.12631) |

Other recent work, noted but not adopted:
- **Routing on LLM prefill activations:** closes 45.6% of the gap to the oracle, but needs an open-weight LLM forward pass ([2603.20895](https://arxiv.org/abs/2603.20895)).
- **Output-length-aware routing (R2-Router):** [2602.02823](https://arxiv.org/abs/2602.02823).
- **AWS IPR:** a production router with **frozen encoders plus per-model adapters**, trained on 1.5M quality-annotated prompts, under 150 ms latency ([2509.06274](https://arxiv.org/abs/2509.06274)).

**Answer to "which beat kNN / embedding+linear, by how much":**
- **Under independent, controlled comparisons, almost none beat it by a meaningful margin.** The gaps are about 0–2.6 points of correctness accuracy or about 0.5–1.5 AUC points ([Li 2025](https://arxiv.org/abs/2505.12601); [EmbedLLM](https://arxiv.org/abs/2410.02223); [UniRoute](https://arxiv.org/abs/2502.08773); [LLMRouterBench](https://arxiv.org/abs/2601.07206)).
- **The one clear exception is the decision objective, not the architecture.** A ranking loss lowered the cost to reach GPT-4 quality from about 94% to 77% on RouterBench ([EquiRouter](https://arxiv.org/abs/2602.03478)).
- **Larger fine-tuned classifiers (BERT, Llama-3-8B) only help once there is enough in-distribution labelled data** ([RouteLLM](https://arxiv.org/abs/2406.18665)).

## Q2. Combining difficulty and category; avoiding collapse

**Structure.** The literature separates two factorisations of "difficulty × category":
- **IRT / MIRT:** P(correct | x, m) = σ(a(x)ᵀ θ_m − b(x)), where b(x) is prompt difficulty, a(x) is which abilities the prompt loads on (the "category"), and θ_m is per-model ability per dimension ([IRT-Router eq. 5](https://arxiv.org/abs/2506.01048)). With 8 models and a few thousand prompts, this has far fewer free parameters than 8 independent heads.
- **Multi-label heads on a shared trunk:** one sigmoid per model, trained with BCE ([CARROT §4.2](https://arxiv.org/abs/2502.03261); [EmbedLLM](https://arxiv.org/abs/2410.02223) is the MF version of this). This is more flexible, but it needs labels for every model.
- **Practical combination:** a shared encoder → (difficulty b, loading vector a) → σ(aᵀθ_m − b + bias_m). It is exactly a multi-label head with a low-rank constraint, and θ_m for a new model can be fitted from a few hundred answers.
  - [tinyBenchmarks](https://arxiv.org/abs/2402.14992) estimates a model's benchmark accuracy within about 2% from about 100 IRT-chosen items per scenario.
  - [UniRoute](https://arxiv.org/abs/2502.08773) does the same with per-cluster error vectors.

**Why routers collapse, and how to prevent it.**

1. **Scalar-score instability.** When the predicted correctness of several models is nearly tied, small errors flip the argmax. As the budget grows, selections then concentrate on the strongest model ([EquiRouter §3](https://arxiv.org/abs/2602.03478)). Lonn shows the mirror image: one *cheap* model wins nearly everywhere, because static aptitudes × complexity rarely outweigh the price term. Mitigations:
   - a ranking-aware loss;
   - reporting a collapse metric (EquiRouter's Routing Collapse Index, or simply the selection histogram);
   - deciding with **calibrated probabilities against an explicit threshold** instead of an argmax of uncalibrated scores.
2. **Cost-aware decision rule.** Pick the *cheapest* model with P̂ ≥ τ, falling back to the max-P̂ model when none qualifies. Or use argmax P̂ − λ·Ĉ, which is RouterBench's λ·P − cost ([2403.12031 §5.1](https://arxiv.org/abs/2403.12031)) and CARROT's plug-in rule. Sweeping τ or λ traces the Pareto/deferral curve, and the operating point is then a product decision.
3. **Calibration.** Plug-in rules are only as good as P̂'s calibration. Fit per-model temperature scaling or isotonic regression on held-out folds, and report Brier, ECE and AUROC per model.
   - Isotonic calibration plus a threshold is cost-optimal under stated assumptions for cascades ([UCCI, 2605.18796](https://arxiv.org/abs/2605.18796)).
   - Laya's own README warns its shipped confidences are "over-confident as shipped" ([Laya README, Calibration](https://huggingface.co/convaiinnovations/laya)).
4. **Edge cases / rare winners.** The residual gap is dominated by queries that only 1–3 models solve ([LLMRouterBench §4.2](https://arxiv.org/abs/2601.07206)). Remedies:
   - an abstain/escalate band: if max P̂ is low, or P̂ is uncertain (e.g. kNN neighbours disagree), route to the strongest model;
   - optionally a cascade with a verifier for formats that can be checked cheaply, such as code tests or `\boxed{}` answers.
   Cascades only beat predictive routing when the judge's error is ≤ ~0.1–0.2 ([RouterBench §5.3](https://arxiv.org/abs/2403.12031)).
5. **Out-of-distribution guard.** If the prompt's nearest-neighbour distance to the training set is large, widen the abstain band. kNN-style methods degrade least under shift ([Li 2025 Table 4](https://arxiv.org/abs/2505.12601)).

## Q3. Backbones

| Backbone | Params | Context | Licence | Notes | Source |
|---|---:|---:|---|---|---|
| ModernBERT-base | 149M | 8,192 | Apache-2.0 | 2T tokens of English + code; no CPU latency on the card | [card](https://huggingface.co/answerdotai/ModernBERT-base) |
| ModernBERT-large | 395M | 8,192 | Apache-2.0 | Laya's English backbone. Lonn measured ~133–143 ms per prompt on an Apple-silicon Mac (Laya forward pass, `legacy/docs/fine_tuned_laya.md`) | [card](https://huggingface.co/answerdotai/ModernBERT-base) |
| Ettin encoder (17M–1B; 150M ≈ base, 400M ≈ large) | 17M … 1B | 8k | MIT | Card claims it beats ModernBERT: GLUE 88.9 vs 88.4 (base), MTEB-v2 English retrieval 45.7 vs 43.9 | [card](https://huggingface.co/jhu-clsp/ettin-encoder-150m) |
| DeBERTa-v3-base | 86M backbone + 98M embeddings | 512 (unverified on card; standard setting) | MIT | 128k vocab; used by RouterDC (mDeBERTa), Hybrid LLM and the NVIDIA classifier | [card](https://huggingface.co/microsoft/deberta-v3-base) |
| DeBERTa-v3-large | 304M + 131M | 512 (unverified) | MIT | | [card](https://huggingface.co/microsoft/deberta-v3-large) |
| gte-modernbert-base (embedder) | 149M | 8,192 | Apache-2.0 | MTEB-en 64.38 (56 tasks) | [card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) |
| bge-base-en-v1.5 (embedder) | ~110M | 512 | MIT | MTEB 63.55 (small 62.17, large 64.23) | [card](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| Qwen3-Embedding-0.6B | 0.6B, 28 layers | 32k | Apache-2.0 | MTEB-Eng-v2 70.70; likely too slow for < 200 ms on CPU (unverified) | [card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |

**CPU latency.** None of these model cards report CPU latency. The only measured number is Lonn's: ModernBERT-large inside Laya at ~140 ms per prompt on an Apple-silicon Mac. A base-size model (149M, about 2.6× fewer parameters) should fit well under 200 ms, but **this is unverified; benchmark it with `legacy/examples/scorer_benchmark.py`-style timing**. kNN or linear heads on top add milliseconds: kNN routing was the fastest method in [Li 2025 Table 3](https://arxiv.org/abs/2505.12601).

**What Laya is** (verified from the installed package `laya==0.3.29` metadata and its checkpoint config):
- **Publisher:** Convai Innovations (author Nandha Kishor M); [HF model](https://huggingface.co/convaiinnovations/laya), [GitHub](https://github.com/NandhaKishorM/laya). Licence Apache-2.0.
- **What it does:** it describes itself as a "non-autoregressive System 1 decision engine". It answers typed questions (choice / score / yes-no) about a text in one forward pass, and is trained with "RLCD" (proper-scoring-rule rewards, a GRPO-style policy gradient).
- **Checkpoints:**
  - `laya`: ModernBERT-large, 421M, 512 tokens. The `rl_agent_config.json` says `encoder: answerdotai/ModernBERT-large`, `max_len: 512`, `head_max_len: 192`.
  - `laya-multilingual`: mmBERT-base, 322M, 1024–8192 tokens.
  - `laya-typed-decisions`.
- **Trainability:** it is open and trainable. `laya.train.TrainConfig` has `freeze_encoder: bool = False`, so even full fine-tuning is supported. Lonn's `legacy/scripts/finetune_laya.py` deliberately freezes the encoder because only 228 labels exist.
- **Jev (TypeSafe AI)** is the proprietary API that Laya's server can emulate (`LAYA_JEV_STRICT`). It is not a model we can train.

**Laya verdict, reasons:**
1. Its output space is the wrong target. Topic and "is it hard?" are *prompt attributes*, and routers that use them land near the dataset-level prior at best ([LLMRouterBench](https://arxiv.org/abs/2601.07206)). Lonn's own runs confirm this: stock Laya, fine-tuned Laya and true topics all score 78.1–78.6%.
2. Per-model correctness would have to be phrased as 8 natural-language questions sharing a 512-token window with the prompt. The question wording is baked into the checkpoint. Retraining needs RLCD plus temperature fitting, and its confidences ship uncalibrated.
3. It is a heavyweight dependency (~1.6 GB for the fine-tuned checkpoint) for something a 149M encoder with an 8-way sigmoid head does directly.

Keep Laya only as a **baseline row** in the evaluation, or as a source of interpretable features in the style of [SeLMRoute](https://arxiv.org/abs/2609.34736) (unverified beyond its abstract).

## Q4. Data efficiency and public data

**How many labels.**
- [EmbedLLM Table 1](https://arxiv.org/abs/2410.02223): correctness accuracy goes 0.637 → 0.708 for kNN and 0.644 → 0.733 for MF between 1k and 5k training questions, and is nearly flat after 10k.
- [RouteLLM](https://arxiv.org/abs/2406.18665): about 1,500 in-domain gold labels (< 2% of the training data) moved MMLU routing from random to +20%. High-capacity classifiers were at random on 65k *out-of-domain* labels.
- [Li 2025 §7](https://arxiv.org/abs/2505.12601): when embeddings have low intrinsic dimension (2–28 measured), kNN needs fewer samples than parametric routers.
- **Implication for Lonn:** about 1–5k *in-distribution* prompts with per-model labels is the useful range. 690 (249 informative) is below it.

**Public per-prompt correctness datasets:**

| Dataset | Size | Models | Licence | Notes | Source |
|---|---|---|---|---|---|
| RouterBench | ~36k prompts, 405k outcomes | 11 (2023-era) | not stated on the HF card (unverified) | MMLU, GSM8K, MBPP, ARC, HellaSwag, Winogrande, MT-Bench, RAG; costs included | [paper](https://arxiv.org/abs/2403.12031), [HF](https://huggingface.co/datasets/withmartian/routerbench) |
| SPROUT | 44,241 prompts (31k/6.6k/6.6k) | 13 columns (paper says 14–15) | not stated (unverified) | GPQA, MMLU-Pro, MATH, MuSR, OpenHermes, RAGBench; zero-shot chat templates; Llama-3.1-70B judge; token counts | [paper](https://arxiv.org/abs/2502.03261), [HF](https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT) |
| EmbedLLM | 112 models × 36,054 questions | 112 open models | Apache-2.0 | MMLU, GPQA, GSM8K, MathQA, LogiQA, …; lm-eval-harness; no costs | [paper](https://arxiv.org/abs/2410.02223), [HF](https://huggingface.co/datasets/RZ412/EmbedLLM) |
| RouterEval | 200M+ records, 12 benchmarks | 8,500+ (Open LLM Leaderboard models) | MIT (repo) | ARC, MMLU(-Pro), GSM8K, IFEval, BBH, GPQA, MATH-L5, MuSR, …; no costs | [paper](https://arxiv.org/abs/2503.10657), [repo](https://github.com/MilkThink-Lab/RouterEval) |
| LLMRouterBench | 400k+ instances, 21 datasets | 33 (incl. GPT-5, Gemini-2.5-pro) | MIT code; data licence not stated | Per-instance score, tokens and cost; includes MATH-500, MBPP, HumanEval, LiveCodeBench, GPQA | [paper](https://arxiv.org/abs/2601.07206), [repo](https://github.com/ynulihao/LLMRouterBench) |

**Transfer to a new pool.** Every public dataset uses *other* models than Lonn's 8. Evidence that transfer works:
- RouteLLM routers trained on GPT-4/Mixtral transferred to Claude-3 Opus/Sonnet and Llama-3.1 70B/8B pairs without retraining (APGR 0.68–0.77 vs 0.5 random). This works because they learn *prompt* hardness, and that transfers ([2406.18665 Table 4](https://arxiv.org/abs/2406.18665)).
- UniRoute routes among 30+ unseen LLMs given a small labelled validation set per new model ([2502.08773](https://arxiv.org/abs/2502.08773)).
- IRT-Router on an unseen model reached only ACC 0.67 / AUC 0.62, so transfer is real but weak ([2506.01048 Table 4](https://arxiv.org/abs/2506.01048)).

Recommended use: **pretrain the prompt side** (the encoder plus the difficulty and loading outputs b(x), a(x)) on the public matrices, which overlap our sources (MMLU-Pro, GPQA, MATH, MBPP, HumanEval, LiveCodeBench, IFEval). Then **fit our 8 models' θ_m and heads on our own bank**. Two cautions:
- These 2024–25 pools are much weaker than ours. Items that are hard for them may be easy for ours, so expect difficulty miscalibration and refit b(x) on our data.
- Check the questions for overlap with our 690 evaluation questions and remove duplicates.

**Dataset-level accuracy as labels.** These give each prompt its source dataset's mean accuracy as a soft target. A model trained on them can only learn "which dataset does this look like", which is the dataset-oracle prior. In-sample, that prior is worth up to about 89–90% at $3.9–5.2 here, but it cannot learn within-dataset differences, and those are exactly the 94%-at-$1.08 headroom.
- [LLMRouterBench](https://arxiv.org/abs/2601.07206) finds that today's routers already sit close to this Dataset Oracle.
- Use these labels as (a) a prior or shrinkage target for a hierarchical model, or (b) a fallback when the prompt is out of distribution.
- Do not use them as primary training targets, and never evaluate with them.

## Q5. Evaluation metrics

- **APGR / CPT(x%)** ([RouteLLM §3](https://arxiv.org/abs/2406.18665)):
  - PGR = (router − weak) / (strong − weak).
  - APGR is the area under PGR vs %-strong-calls.
  - CPT(x%) is the minimum % of strong-model calls that reaches PGR x.
  - These assume a *two-model* setting, so they fit Lonn only in an adapted form.
- **AIQ** ([RouterBench §3](https://arxiv.org/abs/2403.12031)): the average quality over the cost range, taken from the non-decreasing convex hull of the router's cost–quality points as the willingness-to-pay λ is swept (paraphrase; see the paper for the exact normalisation). It is multi-model and cost-aware, so it fits Lonn.
- **Deferral-curve area / Area(50%) / QNC** ([UniRoute §7](https://arxiv.org/abs/2502.08773); EquiRouter's nAUC, QNC and RCI): QNC is the minimum relative cost needed to match the most accurate model.
- **Pareto-front AUC normalised between random and oracle** ([Li 2025 §4](https://arxiv.org/abs/2505.12601)).

**Fit for Lonn's replay.** The replay already has an oracle, the best fixed model, cheapest, random and real billed costs per (question, model). Add:
1. **A deferral curve**: pass rate vs total billed cost over a λ or τ sweep, with AIQ/nAUC normalised so that random = 0 and oracle = 1.
2. **QNC vs Claude Sonnet 5.5**, and the cost at 90% and 95% of Sonnet's pass rate (a CPT analogue).
3. **PGR at the default operating point**, meaning the fraction of the router→oracle gap closed.
4. **Per-model calibration** (Brier, ECE, AUROC of P̂) and **per-dataset breakdown**.
5. **A model-selection histogram plus the RCI**, to catch collapse.
6. **Splits:** k-fold grouped by `prompt_id`, plus leave-one-dataset-out for OOD. All 690 bank questions are also the current evaluation set, so a trained router must be scored by cross-validation or on new questions only.
7. **The 4096-token production variant**, already supported.

## Implications for Lonn (prototype3.0)

**Design.**
1. **`signal` changes** from (topic distribution, complexity) to **`{model_id: P̂(correct)}` + `{model_id: Ĉ(cost)}`**. This ripples through `schemas.py`, `api.py` and the SDK (the CLAUDE.md contract). Optionally keep topic for logs and interpretability.
2. **Predictor v0 (no training of the encoder):**
   - frozen `gte-modernbert-base` or `bge-base-en-v1.5` embeddings;
   - **(a)** kNN (cosine, k tuned by cross-validation) giving per-model mean correctness;
   - **(b)** per-model logistic regression, L2-regularised;
   - **(c)** MIRT, σ(aᵀθ_m − b), with 4–8 dimensions.

   Calibrate each per model (isotonic or Platt) on out-of-fold predictions. Ship whichever wins on cross-validated nAUC, or a simple average. CPU cost is about one embedding pass plus microseconds.
3. **Cost model:** Ĉ_m(x) = input_tokens × in_price + E[output_tokens | dataset-cluster, m] × out_price, with E[…] estimated from the bank's `output_tokens`. This replaces the "equal input/output tokens" list-price assumption.
4. **Decision:** choose the cheapest m with P̂_m ≥ τ, otherwise the argmax of P̂. Use an **abstain band**: escalate to the strongest model when max P̂ < τ_low, or when the kNN distance shows the prompt is out of distribution. τ (or λ) replaces α as the user knob. The fallback list is ordered by the same rule.
5. **Predictor v1 (with more labels):**
   - fine-tune ModernBERT-base or Ettin-150M with a shared trunk, an 8-sigmoid (or MIRT) head, BCE plus a pairwise ranking term in EquiRouter's style;
   - optionally pretrain on SPROUT, EmbedLLM and LLMRouterBench (the prompt side only);
   - train on a rented GPU and run inference on CPU via ONNX or plain PyTorch.

**Labels and data needed.**
- **Grow the answer bank** with new questions from the same nine sources (their unused items) plus realistic chat prompts, for example WildChat, which is already used for Laya labels. The cost is about $0.037 per question across all 8 models (the bank totals about $25.5 for 690 questions), so 5k questions ≈ $185 and 10k ≈ $370, plus Opus judging for writing.
  - **Prioritise informative items**, those where models disagree. Saturated sets (MATH-500, HumanEval+, ARC) add little signal.
  - **These are paid calls:** ask before running, and use `--dry-run`.
- **Keep the existing 690 questions as the frozen test set,** or keep a fixed held-out slice of them. Train only on new questions.
- **Model churn:** when a pool model changes, label about 100–300 informative prompts with the new model and refit only its θ_m or its head (the IRT/UniRoute pattern). Do not retrain everything.
- **Frank assessment.** The current bank is sufficient to *evaluate* routers and to fit low-capacity routers by cross-validation (kNN, logistic, MIRT). It is not sufficient to fine-tune an encoder. Dataset-level labels will reproduce a dataset prior (≤ ~89% here, in-sample), not the per-prompt gains.

**Risks.**
- **Distribution shift:** benchmark prompts are not user prompts, and production uses no format system prompt (`docs/router_evaluation.md`). Correctness labels for open-ended chat need a judge, as WritingBench already does.
- **Overfitting with 249 informative items:** use grouped cross-validation, report confidence intervals, and prefer low-capacity models.
- **Label noise:** temperature-0 single samples. The [Hybrid LLM](https://arxiv.org/abs/2404.14618) results show that response quality is a random variable, so consider 2 samples on borderline items.
- **Calibration drift** when providers change model versions.
- **Price drift:** keep cost estimation separate from the correctness predictor (the CARROT plug-in split) so that prices can change without retraining.

## Sources

- RouterBench: Hu et al., 2024. https://arxiv.org/abs/2403.12031 · data https://huggingface.co/datasets/withmartian/routerbench
- RouteLLM: Ong et al., 2024. https://arxiv.org/abs/2406.18665
- Hybrid LLM: Ding et al., ICLR 2024. https://arxiv.org/abs/2404.14618
- RouterDC: Chen et al., NeurIPS 2024. https://arxiv.org/abs/2409.19886
- EmbedLLM: Zhuang et al., ICLR 2025. https://arxiv.org/abs/2410.02223 · data https://huggingface.co/datasets/RZ412/EmbedLLM
- GraphRouter: Feng et al., ICLR 2025. https://arxiv.org/abs/2410.03834
- IRT-Router: Song et al., ACL 2025. https://arxiv.org/abs/2506.01048
- CARROT / SPROUT: Somerstep et al., 2025. https://arxiv.org/abs/2502.03261 · data https://huggingface.co/datasets/CARROT-LLM-Routing/SPROUT
- MixLLM: Wang et al., NAACL 2025. https://arxiv.org/abs/2502.18482
- UniRoute: Jitkrittum et al., 2025. https://arxiv.org/abs/2502.08773
- RouterEval: Huang et al., 2025. https://arxiv.org/abs/2503.10657 · https://github.com/MilkThink-Lab/RouterEval
- kNN vs learned routers: Li, 2025 (rev. 2026). https://arxiv.org/abs/2505.12601
- Arch-Router: Katanemo, 2025. https://arxiv.org/abs/2506.16655 · https://huggingface.co/katanemo/Arch-Router-1.5B
- Avengers-Pro, 2025. https://arxiv.org/abs/2508.12631
- IPR (AWS), 2025. https://arxiv.org/abs/2509.06274
- LLMRouterBench, 2026. https://arxiv.org/abs/2601.07206 · https://github.com/ynulihao/LLMRouterBench
- EquiRouter / routing collapse, 2026. https://arxiv.org/abs/2602.03478
- R2-Router, 2026. https://arxiv.org/abs/2602.02823
- Prefill-activation router, 2026. https://arxiv.org/abs/2603.20895
- UCCI calibrated cascades, 2026. https://arxiv.org/abs/2605.18796
- SeLMRoute, 2026 (abstract only). https://arxiv.org/abs/2609.34736
- tinyBenchmarks: Polo et al., 2024. https://arxiv.org/abs/2402.14992
- NVIDIA prompt-task-and-complexity-classifier. https://huggingface.co/nvidia/prompt-task-and-complexity-classifier
- ModernBERT. https://huggingface.co/answerdotai/ModernBERT-base · Ettin https://huggingface.co/jhu-clsp/ettin-encoder-150m · DeBERTa-v3 https://huggingface.co/microsoft/deberta-v3-base, https://huggingface.co/microsoft/deberta-v3-large · gte-modernbert-base https://huggingface.co/Alibaba-NLP/gte-modernbert-base · bge-base-en-v1.5 https://huggingface.co/BAAI/bge-base-en-v1.5 · Qwen3-Embedding-0.6B https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Laya: https://huggingface.co/convaiinnovations/laya · https://github.com/NandhaKishorM/laya (facts checked against the installed `laya` 0.3.29 package metadata and the cached checkpoint's `rl_agent_config.json`)
- Lonn internal: `results/router_eval/runs.csv`, `results/benchmark/**`, `docs/router_evaluation.md`, `legacy/docs/fine_tuned_laya.md`
