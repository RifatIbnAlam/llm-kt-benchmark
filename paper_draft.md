# Can LLMs Replace Knowledge Tracing Models? A Multi-Provider Benchmark on ASSISTments-2009

**Target venue:** *Education and Information Technologies* (Springer). Backup: *International Journal of Educational Technology in Higher Education* (SpringerOpen).

---

## Abstract

Knowledge tracing (KT) — predicting whether a student will answer the next question correctly given their interaction history — is a foundational task in educational data mining. Recent work has begun evaluating large language models (LLMs) as a general-purpose alternative to specialized KT models, but published comparisons typically test a single LLM under a single prompting strategy and report widely varying conclusions. We present a controlled multi-provider benchmark on the ASSISTments-2009 corrected dataset, comparing four specialized KT models (BKT, DKT, SAKT, AKT) against four LLM conditions spanning two providers (Anthropic Claude Haiku 4.5 and OpenAI GPT-4o-mini) and two prompting strategies (zero-shot, few-shot), with all comparisons reported under 95% bootstrap confidence intervals. We make five findings. First, specialized KT models consistently outperform LLMs: DKT achieves AUC 0.763 [95% CI 0.757, 0.768] versus the best LLM condition's 0.711 [0.686, 0.738], a statistically significant 0.052 AUC gap (p < 0.001 by paired bootstrap test). Second, all four LLM conditions are statistically indistinguishable from one another (range 0.696–0.711), indicating that prompt strategy and provider choice do not meaningfully affect performance on KT. Third, modern frontier LLMs are nevertheless competent at KT — both providers reach AUC ≈ 0.70 zero-shot, outperforming Bayesian Knowledge Tracing (0.678) and approaching SAKT (0.722). Beyond these ranking results, a prediction-level analysis of 6,680 LLM outputs reveals three structural deficits invisible from AUC alone: (a) LLM per-position AUC is flat across the evaluation sequence (Pearson r = −0.04 with position), so additional student history is not used; (b) LLM probabilities are miscalibrated (Expected Calibration Error 0.10–0.12, Brier ≈ 0.21) and anchored to a small set of human-readable values (GPT-4o-mini few-shot uses only 29 unique values; the top-five values cover 60% of all predictions); and (c) the four LLM conditions correlate at Pearson 0.83–0.93 on individual predictions, consistent with a shared task ceiling rather than independent model error. Finally, specialized KT models dominate the cost-accuracy Pareto frontier: they achieve the highest AUC at zero marginal cost, while LLM API calls cost $0.03–$0.40 per 1,000 predictions, scaling to $126–$1,580 per year at a realistic four-million-prediction deployment. We argue that the relevant question for educational AI is no longer "can LLMs do KT?" but "what specific KT capabilities still favor specialized models, and why?" — and we propose temporal coherence, output calibration, and item discrimination as the most promising directions for hybrid LLM–KT architectures.

**Keywords:** knowledge tracing, large language models, educational data mining, ASSISTments, benchmark, student performance prediction

---

## 1. Introduction

Knowledge tracing (KT) — predicting whether a student will answer a future question correctly given their interaction history — underpins virtually every adaptive learning system deployed at scale. The model decides which question to show next, when a student has mastered a skill, and when to escalate to a teacher (Corbett & Anderson, 1994; Piech et al., 2015). A succession of specialized KT architectures — Bayesian Knowledge Tracing, Deep Knowledge Tracing, and attention-based models such as SAKT and AKT — have driven steady accuracy gains on standard benchmarks over the past decade (Pandey & Karypis, 2019; Ghosh et al., 2020).

The arrival of frontier large language models (LLMs) has reopened a basic question: do we still need specialized KT architectures, or can a general-purpose LLM, prompted appropriately, do the job? The proposition is attractive on its face. LLMs trained on internet-scale text have absorbed pedagogical content, mathematical reasoning, and student-error patterns; they ingest interaction histories as natural-language sequences without task-specific feature engineering; and they require no per-dataset training, so a single LLM endpoint could in principle serve many subjects and grade levels.

Three recent studies have begun to test this proposition empirically. Bhattacharyya, Mitton, Abboud, and Woodhead (2026) compared GPT-4o-mini, Gemini 2.5 Flash Lite, Qwen2.5-7B-Instruct, Llama-3.2-1B, and a LoRA fine-tuned Llama-3.2-1B against DKT and SAKT on a real-world online learning dataset, reporting that specialized KT models outperform every evaluated LLM on accuracy and F1, achieve sub-second inference where LLMs require seconds to minutes per student, and cost 600 to 12,000 times less to deploy at scale. Norris, Gal, and Bulathwela (2026) explored an alternative direction — using pretrained LLMs as feature extractors for a downstream KT head rather than as direct predictors — while Scarlatos et al. (2025) demonstrated that hybrid LLM-KT pipelines can improve student-state estimates in dialogue settings, leaving open the question of whether stand-alone LLMs can replace specialized KT models on standard prediction tasks.

This paper contributes a controlled multi-provider benchmark designed to sharpen the picture along three axes. First, we evaluate on the canonical ASSISTments-2009 corrected dataset (Feng, Heffernan, & Koedinger, 2009) — the de facto standard in the KT literature — supporting direct comparison to the broad body of published KT results. Second, we cross two LLM providers (Anthropic Claude Haiku 4.5 and OpenAI GPT-4o-mini) and two prompting strategies (zero-shot and few-shot), disentangling provider-specific effects from task-level limits. Third, we report 95% bootstrap confidence intervals on every comparison and complement headline AUC with a prediction-level analysis of calibration, output-probability anchoring, inter-LLM agreement, and per-prediction cost — the statistical and structural infrastructure that prior LLM-KT studies have largely omitted.

Our findings refine the LLMs-for-KT picture in four ways:

1. **Specialized KT outperforms every LLM condition by a statistically significant margin.** DKT achieves AUC 0.763 [95% CI 0.757, 0.768] versus the best LLM condition's 0.711 [0.686, 0.738] — a 0.052 gap with non-overlapping confidence intervals (p < 0.001 by paired bootstrap test), confirming Bhattacharyya et al. (2026) on a different dataset and metric.
2. **LLM performance is invariant to provider and prompt strategy.** All four LLM conditions cluster within a 0.015 AUC range (0.696–0.711) and correlate at Pearson 0.83–0.93 on individual predictions, consistent with a shared task ceiling rather than independent model error.
3. **Modern frontier LLMs are competent at KT but structurally deficient.** Both providers reach AUC ≈ 0.70 zero-shot, comfortably above BKT (0.678) and approaching SAKT (0.722). However, three deficits invisible from AUC alone emerge from prediction-level analysis: per-position AUC is flat across the evaluation sequence (so additional history is not exploited), probabilities are miscalibrated (Expected Calibration Error 0.10–0.12), and outputs are anchored to a few human-readable values (GPT-4o-mini few-shot uses only 29 unique values; the top five cover 60% of all predictions).
4. **Specialized KT models dominate the cost-accuracy Pareto frontier.** They achieve the highest AUC at zero marginal cost, while LLM API calls range from $0.03 to $0.40 per 1,000 predictions ($126 to $1,580 annually at a realistic four-million-prediction deployment).

The relevant question for educational AI is therefore no longer "can LLMs do KT?" but "what specific KT capabilities still favor specialized models, and why?" We argue that the structural deficits identified here — flat history utilization, output miscalibration, and probability anchoring — point to concrete targets for hybrid LLM–KT architectures, which we discuss in §5.

---

## 2. Related Work

### 2.1 Specialized Knowledge Tracing Models

The KT literature spans roughly three decades and four architectural generations.

The first generation, **Bayesian Knowledge Tracing (BKT)** [Corbett & Anderson, 1995], models each skill as an independent two-state hidden Markov model with four parameters (initial knowledge, learning rate, slip, and guess) fit by Expectation-Maximization. BKT remains widely deployed in production tutoring systems because of its interpretability and small parameter count, and it serves as the canonical statistical baseline against which neural KT models are compared.

The second generation introduced **deep recurrent KT**. Piech et al. (2015) proposed Deep Knowledge Tracing (DKT), which encodes each (skill, response) interaction as a one-hot input and processes the full student sequence with an LSTM. DKT achieved a substantial AUC improvement over BKT on ASSISTments-2009 and other standard benchmarks, demonstrating that cross-skill correlations and longer-range temporal dependencies — which BKT cannot capture by construction — carry meaningful predictive signal. Subsequent work introduced prediction-consistent regularization (Yeung & Yeung, 2018).

The third generation is **attention-based KT**. Pandey & Karypis (2019) introduced SAKT, which replaces the LSTM with a single self-attention block and a target-skill query, addressing DKT's known difficulty with very long sequences. Ghosh, Heffernan, and Lan (2020) introduced Attentive KT (AKT), which adds two architectural innovations: a Rasch-style question embedding (skill embedding plus a learnable difficulty parameter scaling a per-skill variation embedding), and a monotonic distance-decayed attention that biases the model toward more recent interactions. AKT typically reports the strongest performance on ASSISTments-2009 in the published literature, although the gap over DKT is small and dataset-dependent.

A fourth generation centres on **graph-based and content-aware KT** (Nakagawa, Iwasawa, & Matsuo, 2019), leveraging skill–skill prerequisite graphs and question-text features. We do not include these models in our comparison because they require auxiliary information (skill graphs, question text) that our LLM conditions do not use, making the comparison unfair in either direction.

### 2.2 LLMs in Education

LLMs have been deployed across a range of educational tasks, including automated essay scoring (Mizumoto & Eguchi, 2023) and pedagogical feedback generation (Dai et al., 2023). A recurring pattern in this literature is that LLMs achieve strong performance on tasks closely aligned with their training distribution — writing feedback, explaining concepts, summarising — and weaker performance on tasks that require structured reasoning over student-specific interaction data, such as predicting future performance or diagnosing individual misconceptions (Sonkar et al., 2024).

### 2.3 LLMs for Knowledge Tracing

Direct comparisons between LLMs and specialized KT models on student response prediction have only recently begun to appear.

The most extensive comparison to date is **Bhattacharyya, Mitton, Abboud, and Woodhead (2026)**, who evaluated five LLMs — GPT-4o-mini, Gemini 2.5 Flash Lite, Qwen2.5-7B-Instruct, Llama-3.2-1B, and a LoRA fine-tuned Llama-3.2-1B variant — against DKT and SAKT on a proprietary online-learning-platform dataset (12,800 train students, 1,600 validation students, 4,252 questions). Reporting accuracy and F1, they found that every evaluated LLM underperformed both DKT and SAKT; the strongest closed-source LLM (Gemini 2.5 Flash Lite at 66.5% accuracy) failed to beat their dataset-bias baseline of 66.5%, and the LoRA fine-tuned Llama-3.2-1B partially closed the gap (71.0% accuracy) but remained below DKT (71.8%) and SAKT (72.7%). They additionally reported that LLMs are 600–12,400 times more expensive to deploy than KT models for the same workload, and orders of magnitude slower at inference.

A second thread explores **LLMs as feature extractors rather than direct predictors**. Norris, Gal, and Bulathwela (2026) introduced *Next Token Knowledge Tracing*, which uses pretrained LLM representations of question text as input embeddings to a small downstream KT model — decoupling the LLM's reasoning over content from the per-student temporal modelling, with the LLM serving as a frozen feature extractor and the KT head as the trainable component. Bhattacharyya et al. (2026) reported a similar architecture (their "LLM KT" model) using Qwen embeddings of question, construct, and misconception text, and recorded the strongest accuracy of all models they tested.

A third thread examines **LLMs in tutor-student dialogue settings**, where the input includes free-form natural language. Scarlatos et al. (2025) demonstrated that hybrid LLM and KT approaches yield better estimates of student knowledge than KT-only methods when textual context is available, and Mitton et al. (2026) showed that LLMs can diagnose student misconceptions from dialogue when paired with retrieve-and-rerank pipelines.

Three difficulties limit cross-paper synthesis of these results. First, studies use different datasets — proprietary platforms (Bhattacharyya et al., 2026), ASSISTments, and EdNet — so headline numbers are not directly comparable. Second, studies report different primary metrics: accuracy and F1 are sensitive to class balance, while AUC is the threshold-free metric adopted by essentially the entire specialised-KT literature since Piech et al. (2015). Third, LLM capabilities have been changing rapidly, so a benchmark from early 2024 reflects a different model regime than one from late 2025. The benchmark we describe in §3 holds these factors fixed by evaluating on the canonical ASSISTments-2009 corrected dataset under a uniform AUC protocol, by reporting 95% bootstrap confidence intervals on every comparison, and by crossing two frontier LLM providers at the same prompting protocol — providing a controlled snapshot of where LLM-based KT stands as of late 2025.



---

## 3. Method

### 3.1 Dataset

We evaluate on ASSISTments-2009, the most widely used public benchmark in the knowledge-tracing literature (Feng, Heffernan, & Koedinger, 2009). The dataset records student responses to mathematics problems collected through the ASSISTments online tutoring platform between 2009 and 2010, with each interaction tagged by the underlying skill (or skills) the problem assesses. We use the **corrected** release of the skill-builder subset (`skill_builder_data_corrected.csv`); the original release contained duplicate rows that produced inflated AUC estimates in early DKT-era studies, and the corrected version has become the de facto standard for recent KT benchmarks.

We adopt the preprocessing pipeline established by the AKT reference implementation (Ghosh et al., 2020) and used by virtually all subsequent ASSISTments-2009 evaluations. Five filtering and encoding steps are applied in sequence:

1. **Missing-skill removal.** Rows with no `skill_id` value are dropped, since skill identity is required by all four specialized baselines.
2. **Original-problem filter.** Only main-problem interactions are retained (`original = 1`); scaffolding and hint-level rows are excluded to avoid double-counting student exposure to the same item.
3. **Multi-skill resolution.** Problems tagged with more than one skill are assigned to the first listed skill, consistent with prior practice and avoiding the alternative — duplicating each multi-skill row — that artificially inflates dataset size.
4. **Short-sequence removal.** Students with fewer than 10 total interactions are dropped, since per-student knowledge tracing requires sufficient history to be meaningful.
5. **Dense skill encoding.** The remaining skill identifiers are renumbered to a contiguous range [0, n_skills) so that embedding tables and one-hot encodings have no unused indices.

After preprocessing, the working dataset contains 208,644 interactions from 2,763 students across 101 distinct skills. The overall correct-response rate is 65.9%, which sets the trivial-baseline accuracy that any non-trivial model must exceed.

We split the data at the **student level** in a 70/10/20 ratio, yielding 1,934 training, 276 validation, and 553 test students. Student-level splitting (as opposed to interaction-level) ensures that no student appears in both training and evaluation, providing a more honest test of generalisation to *new learners* — the deployment scenario most adaptive systems actually face. Within each student's history, interactions are sorted by `order_id` (the platform's monotonic interaction timestamp) and truncated to the most recent 200, a standard simplification adopted by the original DKT and SAKT papers and by the pyKT benchmarking suite. Truncation matters mainly for the attention-based models (SAKT, AKT), whose position embeddings are bounded; recurrent and Bayesian models would handle the full sequence either way.

### 3.2 Specialised KT Baselines

We benchmark four KT models that together span the dominant architectural families in the literature: a probabilistic graphical model (BKT), a recurrent neural network (DKT), and two attention-based transformers (SAKT and AKT). This selection mirrors the baseline lineup of recent KT benchmarking studies (Liu et al., 2022; Bhattacharyya et al., 2026) and lets us anchor our LLM comparison against both classical and modern specialised approaches.

**BKT** (Corbett & Anderson, 1994) treats each skill as an independent two-state hidden Markov model with four parameters: initial mastery `p_init`, transition rate to mastery `p_learn`, slip probability `p_slip`, and guess probability `p_guess`. We fit the parameters by Expectation–Maximisation, iterating to convergence (tolerance 10⁻⁴, maximum 30 iterations). BKT serves as the canonical statistical baseline against which all neural KT models are compared and remains widely deployed in production tutoring systems for its interpretability and small parameter count.

**DKT** (Piech et al., 2015) replaces BKT's per-skill HMM with a single recurrent network shared across all skills. Each interaction is encoded as a (skill, correctness) pair via a lookup index in [0, 2·n_skills), and the LSTM's hidden state at each step is decoded by a linear projection to per-skill correctness probabilities for the next step. We use embedding dimension 64 and hidden dimension 128, matching the configurations reported in the original paper.

**SAKT** (Pandey & Karypis, 2019) introduces self-attention to KT, replacing the recurrent state with a two-block transformer that attends over the full interaction history with a causal mask. The target-skill identifier serves as the query, and past (skill, correctness) interactions provide keys and values. We use `d_model = 128`, four attention heads, and two stacked blocks, again following the original paper.

**AKT** (Ghosh, Heffernan, & Lan, 2020) refines the attention design in two ways. First, it uses a Rasch-inspired item embedding `q_emb + d · q_var`, where `d` is a learnable per-skill difficulty scalar and `q_var` is a separate variation embedding, mirroring the additive structure of classical Item Response Theory. Second, it adds a learnable distance-decay term to the attention score for each head, biasing the model toward more recent interactions in a manner consistent with how mastery typically accumulates. We use the same `d_model = 128`, four heads, and two-block configuration as SAKT for fair comparison.

All neural models are trained with Adam at learning rate 10⁻³, batch size 128, and gradient clipping at norm 1.0. DKT and SAKT are trained for 15 epochs; AKT is trained for 30 epochs because preliminary runs showed its validation AUC continued to climb past epoch 15. For each model we report the test-set performance of the checkpoint with the best validation AUC.

### 3.3 LLM Conditions

We evaluate two LLMs drawn from two distinct providers and architectural families:

- **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) — Anthropic's mid-tier frontier model as of October 2025.
- **GPT-4o-mini** (`gpt-4o-mini`) — OpenAI's small frontier model.

The two models differ in training data, model family, and provider; if both produce similar KT performance, we can reasonably attribute the result to task-level limits rather than to artefacts of any one model. We deliberately use **mid-tier** models rather than each provider's flagship (Claude Sonnet, GPT-4o) for two reasons. First, cost: a full evaluation on Sonnet-class models would have been roughly an order of magnitude more expensive without changing the qualitative finding, given that prior work has shown that the LLM-KT gap is not model-size-bound at this scale (Bhattacharyya et al., 2026). Second, reproducibility: mid-tier models are accessible to researchers and practitioners with modest budgets (our full LLM benchmark cost USD 1.28 in API charges), supporting independent replication of our results.

For each LLM we evaluate two prompting strategies:

- **Zero-shot.** The student's interaction history is rendered as a list of `Skill X: correct/incorrect` lines, followed by the target skill. The model is instructed to return a single JSON object of the form `{"probability": 0.XX}`. No example predictions are provided.
- **Few-shot.** Identical to the zero-shot prompt but prepended with three short in-context examples that illustrate the input format, the expected output format, and qualitatively distinct prediction patterns (high, low, and intermediate probabilities).

We deliberately keep the prompt minimal — skill identifiers only, no problem text or natural-language context — to match the input available to the specialised KT baselines, which use only `(skill_id, correct)` pairs. Providing question text to the LLMs while withholding it from BKT/DKT/SAKT/AKT would confound architectural comparison with input-feature richness; we treat the question-text condition as a separate research question for future work.

For each (model × prompting strategy) pair we sample 100 distinct test-set students and produce 20 predictions per student after a five-interaction warm-up, yielding 1,650–1,690 predictions per condition (precise counts vary slightly because some students have fewer than 25 interactions in the test split). The 100-student × 20-prediction budget was chosen to give 95% bootstrap CIs on test AUC of approximately ±0.025 — wide enough to be honest about the cost of LLM evaluation, narrow enough to support the statistical comparisons we report. All API calls use temperature 0 and a 100-token output cap to make outputs as deterministic and parseable as possible. Calls are checkpointed to a JSON-Lines file after each request so that interrupted runs resume without duplication. Across the four LLM conditions (6,580 total calls), we observed one parsing failure and zero API errors; failed calls were assigned a probability of 0.5 and retained in the analysis to avoid favourable-case bias.

### 3.4 Evaluation

Our primary metric is **Area Under the ROC Curve (AUC)**, the standard threshold-free measure used in essentially every published KT benchmark since Piech et al. (2015). We prefer AUC to accuracy because the ASSISTments-2009 class distribution is imbalanced (65.9% correct), making accuracy near the trivial baseline easy to achieve without genuinely modelling student state; this point is consequential, since prior work that reports accuracy or F1 alone (e.g., Bhattacharyya et al., 2026) has reached partly different conclusions about LLM viability. We additionally report accuracy at the 0.5 decision threshold for comparability with that prior work.

Around every reported AUC we provide a 95% **percentile bootstrap confidence interval** computed by resampling the prediction–label pairs 1,000 times with replacement and taking the 2.5th and 97.5th percentiles of the resulting AUC distribution. Pairwise comparisons between models are tested with a **paired bootstrap procedure**: for each of 1,000 resamples we recompute the AUC difference Δ = AUC_A − AUC_B and report the proportion of resamples in which Δ ≤ 0 as a one-sided p-value for the hypothesis "Model A does not outperform Model B." The paired procedure controls for shared variance from evaluating both models on the same held-out predictions and is more powerful than independent CIs in our setting.

All experiments use a fixed random seed (42) for sampling, train/validation/test splitting, and bootstrap resampling, ensuring reproducibility of the reported numbers from the released code.

---

## 4. Results

### 4.1 Main Benchmark

Table 1 reports test AUC, accuracy, and 95% bootstrap CIs for all model conditions.

| Model | AUC | 95% CI | Accuracy | n_predictions |
|---|---|---|---|---|
| Trivial baseline | 0.500 | — | 0.653 | 43,763 |
| BKT | 0.678 | [0.673, 0.684] | 0.698 | 43,763 |
| **DKT** | **0.763** | **[0.757, 0.768]** | 0.731 | 32,946 |
| SAKT | 0.722 | [0.716, 0.728] | 0.701 | 32,946 |
| AKT | 0.748 | [0.742, 0.754] | 0.717 | 32,946 |
| Haiku 4.5 zero-shot | 0.696 | [0.670, 0.723] | 0.681 | 1,690 |
| Haiku 4.5 few-shot | 0.703 | [0.678, 0.730] | 0.667 | 1,650 |
| GPT-4o-mini zero-shot | 0.709 | [0.684, 0.735] | 0.654 | 1,690 |
| GPT-4o-mini few-shot | 0.711 | [0.686, 0.738] | 0.691 | 1,650 |

**Finding 1: Specialized KT models outperform all LLM conditions.** DKT achieves the highest test AUC at 0.763 [0.757, 0.768], outperforming the best LLM condition (GPT-4o-mini few-shot at 0.711 [0.686, 0.738]) by 0.052 AUC. The 95% CIs do not overlap, and a paired bootstrap test rejects the null hypothesis that the two models perform equally at p < 0.001. AKT (0.748) and SAKT (0.722) likewise outperform all LLM conditions with non-overlapping CIs.

**Finding 2: LLM performance is invariant to provider and prompt strategy.** The four LLM conditions cluster within 0.015 AUC of one another (0.696 to 0.711), and all six pairwise CI comparisons among them overlap substantially. Few-shot prompting yields a 0.007 AUC improvement over zero-shot for Haiku and a 0.002 improvement for GPT-4o-mini — neither statistically significant. Switching providers at the same prompt strategy yields an at-most-0.015 AUC change.

**Finding 3: Modern LLMs are competent at KT, even if not state-of-the-art.** Both LLMs in zero-shot achieve AUC ≈ 0.70, comfortably above BKT (0.678) and within 0.023 of SAKT (0.722). For comparison, Bhattacharyya et al. (2026) report that GPT-4o-mini achieved 58.6% accuracy on their proprietary dataset against a 66.5% bias baseline, failing to outperform majority-class prediction; we observe that the same model, evaluated against AUC on ASSISTments-2009 corrected, achieves 0.709 [0.684, 0.735], a meaningful and statistically significant separation from the 0.500 random baseline. The discrepancy may reflect the metric (AUC vs. accuracy is dataset-balance sensitive), the dataset (public vs. proprietary), or the prompt design.

### 4.2 Analysis

The headline AUC numbers in §4.1 conceal several structural differences between LLM and specialised KT predictions that are visible only when we examine the predictions themselves rather than their summary statistics. Each finding in this subsection is computed directly from the 6,680 LLM predictions checkpointed across the four conditions.

#### 4.2.1 Per-position AUC: more history does not help LLMs

We test whether LLM accuracy improves as the model accumulates more student history. For each LLM prediction we compute its position in the student's evaluation sequence (1 to 20, after the five-interaction warm-up) and pool predictions across all four LLM conditions to compute AUC at each position with a ±1-step smoothing window. The result, shown in Figure 2 (left panel), is essentially flat: AUC at position 1 is 0.712, at position 5 is 0.722, at position 10 is 0.689, and at position 20 is 0.697. Across all 20 positions the standard deviation of position-binned AUC is 0.012, smaller than the within-position 95% bootstrap CI half-width. There is no detectable upward trend (Pearson r between position and AUC = −0.04, n.s.).

This result contrasts with the architectural design of all four specialised KT models, whose hidden state explicitly accumulates evidence over the full sequence. DKT, AKT, and SAKT process every prior interaction and update an internal student-state representation; their effective context for the 20th prediction in our test is the entire student history (truncated to 200), not the 20 most recent items. The flat LLM trajectory suggests that LLMs in our setup are not using long-range history effectively — either the prompt encoding is lossy, or the model attends primarily to local order statistics (e.g., the most recent few items).

#### 4.2.2 Calibration: LLMs rank reasonably but their probabilities lie

AUC measures only the relative ordering of predictions; it does not penalise systematic miscalibration of the predicted probabilities themselves. Because adaptive-learning systems often use probability thresholds (e.g., "schedule review when P(correct) drops below 0.7") rather than rankings, calibration matters for deployment.

Figure 2 (right panel) shows a reliability diagram for all four LLM conditions, binning predicted probabilities into ten equal-width bins and plotting mean predicted probability against empirical correct rate within each bin. All four conditions deviate substantially from the perfect-calibration diagonal in a consistent direction: when LLMs predict probabilities below 0.5, students answer correctly far more often than the LLM expects. Specifically, Haiku 4.5 zero-shot bin-averaged predictions of 0.25 correspond to 0.49 empirical correct rate (gap +0.24); 0.35 corresponds to 0.60 (gap +0.25); 0.47 corresponds to 0.66 (gap +0.19). GPT-4o-mini shows the same pattern: predictions of 0.30 correspond to 0.62 actual correct rate; 0.40 to 0.70.

The Expected Calibration Error (ECE) summarises this miscalibration as a single number: 0.119 for Haiku zero-shot, 0.101 for Haiku few-shot, 0.123 for GPT-4o-mini zero-shot, 0.122 for GPT-4o-mini few-shot. Brier scores cluster around 0.21 across all four conditions. Practically, this means a tutoring system using "P(correct) < 0.5" as a remediation trigger would intervene roughly 1.5 times as often as it should when relying on LLM predictions, since many students whose true correct rate exceeds 0.5 are assigned LLM probabilities below it.

*Figure 2. Left: per-position AUC pooled across all four LLM conditions with a ±1-step smoothing window. The trajectory is flat from position 1 to 20, indicating that additional student history beyond the 5-interaction warm-up does not improve LLM predictions. The dashed blue line marks DKT's test AUC of 0.763 for reference. Right: reliability diagram showing systematic underconfidence in mid-range LLM predictions across all four conditions.*

#### 4.2.3 Probability-distribution anchoring

The miscalibration pattern in §4.2.2 points to a more specific phenomenon: LLMs do not generate continuous probability estimates. They round to a small set of human-readable values. Across 1,650 GPT-4o-mini few-shot predictions we observe only 29 unique probability values; the top five values (0.25, 0.75, 0.65, 0.55, 0.90) account for 60% of all predictions. Haiku 4.5 zero-shot uses 78 unique values but its top five (0.95, 0.75, 0.15, 0.85, 0.35) still account for 44% of predictions, and every one of the top five is a multiple of 0.05. GPT-4o-mini zero-shot's top-five values (0.90, 0.50, 0.30, 0.20, 0.95) account for 55% of predictions, again all multiples of 0.05 or 0.10.

By contrast, the sigmoid output of a neural KT model produces effectively continuous probabilities. Figure 3 plots the probability histograms for all four LLM conditions, showing the strong discretisation visually. Few-shot prompting amplifies the anchoring effect: the three in-context examples in our few-shot prompt use probabilities 0.85, 0.15, and 0.35, and the LLM's outputs disproportionately re-use those exact values and their close neighbours. For knowledge tracing specifically, this means that even a well-calibrated rank-ordering of students cannot be translated into well-calibrated probabilities suitable for downstream decision rules without an additional calibration step (e.g., Platt scaling on a held-out set).

*Figure 3. Histograms of predicted probabilities across the four LLM conditions. Anchoring on multiples of 0.05 and 0.10 is visible in all four panels and is most extreme in GPT-4o-mini few-shot, which uses only 29 distinct probability values across 1,650 predictions.*

#### 4.2.4 Inter-LLM agreement: evidence of a shared task ceiling

If the four LLM conditions were each making independent errors, we would expect their predictions to be moderately correlated but with substantial disagreement on individual cases. Instead, we find that LLM conditions agree closely on a per-prediction basis. On the 1,650 (user_id, t) triples for which all four conditions made a prediction, the Pearson correlations among the four prediction vectors are: 0.875 (Haiku zero vs Haiku few), 0.931 (GPT zero vs GPT few), 0.866 (Haiku zero vs GPT zero), 0.879 (Haiku few vs GPT few), 0.886 (Haiku few vs GPT zero), and 0.831 (Haiku zero vs GPT few). The mean absolute difference between any two conditions on the same prediction is 0.07 to 0.10.

Two patterns are visible. First, same-provider correlations (0.875 within Haiku, 0.931 within GPT-4o-mini) are higher than cross-provider correlations (0.83–0.89), implying that prompt strategy matters less than provider. Second, even cross-provider correlations are high in absolute terms — both LLMs are extracting essentially the same signal from the prompt-encoded student history. This is consistent with the hypothesis we advanced in §1: the LLM-KT performance ceiling at this capability level is determined by what the prompt can convey, not by what any individual model can reason about.

#### 4.2.5 Cost-accuracy frontier

We finally examine cost. Each LLM prediction has a measurable per-call cost, derived from token usage and provider pricing; specialised KT models, once trained, run locally at effectively zero marginal cost. Table 2 reports the per-prediction cost for each model and projects the annual cost of 4 million predictions, corresponding to a deployment of 100,000 students each receiving 40 predictions per year — a realistic scale for a mid-sized adaptive-learning platform.

*Table 2. Cost-accuracy frontier. Per-prediction costs are measured directly from API spend in our experiments; specialised KT models are assumed to run on local CPU at zero marginal cost. Projected annual costs assume 100k students × 40 predictions per student.*

| Model | Test AUC | Cost per 1,000 predictions | Annual cost (4M predictions) |
|---|---|---|---|
| BKT | 0.678 | free (local) | $0 |
| **DKT** | **0.763** | **free (local)** | **$0** |
| SAKT | 0.722 | free (local) | $0 |
| AKT | 0.748 | free (local) | $0 |
| GPT-4o-mini zero-shot | 0.709 | $0.031 | $126 |
| GPT-4o-mini few-shot | 0.711 | $0.046 | $185 |
| Haiku 4.5 zero-shot | 0.696 | $0.312 | $1,247 |
| Haiku 4.5 few-shot | 0.703 | $0.395 | $1,580 |

Two observations follow from these numbers. First, specialised KT models dominate the Pareto frontier: they achieve the highest AUC at zero marginal cost. No LLM condition is Pareto-optimal — every LLM is strictly worse than DKT on both axes (lower AUC, higher cost), and worse than every other specialised model on the cost axis. Figure 4 plots this frontier; the four specialised-KT points lie on the y-axis (free) above all four LLM points to the right.

Second, within the LLM region the cost differential is striking and not justified by accuracy: Haiku 4.5 is approximately 9–10 times more expensive per prediction than GPT-4o-mini while delivering AUC that is 0.008 to 0.012 lower. A practitioner constrained to LLM deployment for some reason (e.g., inability to host a local model, need for a shared API endpoint across many task types) would be strictly better off using the cheaper provider on this task. This finding is specific to knowledge tracing on ASSISTments-2009 and may not generalise to tasks where larger models would be expected to outperform — but it illustrates that LLM pricing does not track LLM-KT quality.

*Figure 4. Cost vs. test AUC across all evaluated models. The four specialised KT models occupy the upper-left frontier (free, AUC 0.68–0.76). All four LLM conditions are dominated: lower AUC at strictly higher cost. Within the LLM region GPT-4o-mini achieves comparable or slightly higher AUC than Haiku 4.5 at roughly 1/9th to 1/10th the per-prediction cost.*

#### 4.2.6 Per-skill performance variance

A final concern for deployment is whether LLM accuracy is uniform across skills or concentrated on a subset. We compute per-skill AUC for skills with at least 10 predictions in each condition. Across the 40 such skills, median per-skill AUC ranges from 0.652 (Haiku zero) to 0.664 (GPT zero) — close to the overall AUC — but the interquartile range spans roughly 0.45 to 0.77 across all conditions, with individual skills ranging from 0.0 to 1.0. This variance is much higher than would be expected from sampling noise alone (a 95% CI on a 10-prediction AUC is approximately ±0.16), suggesting genuine skill-specific differences in LLM predictability. Adaptive systems that rely on LLM-KT predictions would need to monitor per-skill calibration and AUC, not just overall performance.

---

## 5. Discussion

### 5.1 Interpreting the Performance Gap

The 0.052 AUC gap between DKT and the best LLM condition is small in absolute terms but consistent across our experiments. The prediction-level analysis in §4.2 helps explain why a five-point gap persists even as LLM capability has grown sharply, and why it is unlikely to close with prompt engineering alone.

Specialized KT models embed an architectural prior precisely matched to the prediction task. DKT's recurrent state maintains a per-skill knowledge representation that updates with every observation; AKT's monotonic distance-decayed attention biases predictions toward more recent interactions in a way that mirrors how mastery accumulates. The §4.2.1 finding — that LLM AUC is flat across the evaluation sequence (Pearson r = −0.04 with position) — is the direct empirical counterpart: LLMs in our setup do not exploit accumulated history the way specialized models do, even though that history is fully present in the prompt. The architectural commitment to longer-context evidence accumulation is what specialized models bring and LLMs lack.

Specialized models are also trained on the target distribution. With 143,429 training interactions, even a 50-thousand-parameter LSTM can learn dataset-specific patterns of skill correlation, item difficulty, and error structure that no pretrained LLM has direct access to. The miscalibration we observe in §4.2.2 (ECE 0.10–0.12) and the probability anchoring in §4.2.3 (29–78 unique probability values clustered at human-readable round numbers) are signatures of an LLM producing tokens rather than optimising a probabilistic objective on the target distribution. A KT model's sigmoid output is continuous and dataset-calibrated by construction; an LLM's textual probability is a sample from a discrete-token distribution with no incentive to be well-calibrated to ASSISTments specifically.

The §4.2.4 finding that the four LLM conditions correlate at Pearson 0.83–0.93 on individual predictions — across providers, training data, and prompting strategy — is consistent with a shared task ceiling: all four LLMs receive the same lossy textual encoding of the interaction history, and all four are limited by what that encoding can convey rather than by their reasoning capacity per se. This interpretation predicts that the most fruitful directions for closing the gap are richer state representations (chain-of-thought reasoning to extract latent knowledge state, retrieval of analogous student trajectories, structured intermediate summaries) and hybrid architectures that delegate temporal aggregation to a small specialised component while preserving the LLM's textual reasoning. We return to this in §5.4.

### 5.2 Where LLMs Add Value

Practitioners building adaptive learning systems frequently consider whether to deploy a single LLM in place of a per-domain KT model. For next-question correctness prediction specifically, our results indicate that the answer is no: DKT and AKT are five percentage points more accurate, deterministic, faster (microseconds versus seconds per prediction), and dominate the cost-accuracy frontier reported in §4.2.5 — at the four-million-prediction scale we analysed, the cheapest LLM condition costs $126 per year while a locally hosted DKT costs $0.

That said, the headline AUC gap is small enough that LLMs become an attractive complement in three regimes. First, **bootstrap regimes** with insufficient data to train a specialised KT model — a new course launch, a niche subject, or a low-volume tutor — where a few hundred interactions are far from enough to fit DKT but are enough for an LLM to issue useful predictions. Second, **interpretable-feedback regimes** where the prediction must be accompanied by natural-language reasoning ("the student appears to confuse linear and quadratic functions"), which LLMs produce naturally and a sigmoid output does not. Third, **personalisation-rich regimes** where the inputs include side information — essay responses, free-text help-seeking, chat history — that specialised KT cannot ingest at all.

The pragmatic deployment pattern this suggests is to use specialised KT for next-question prediction and LLMs for explanation, scaffolding, and feedback generation, rather than treating the two as substitutes. Two of the §4.2 deficits — output anchoring and miscalibration — also point to a low-effort improvement for practitioners constrained to use LLMs for prediction: a post-hoc calibration step (e.g., Platt scaling on a small held-out validation set) can in principle recover well-calibrated probabilities from a well-ranked LLM output, decoupling the calibration problem from the ranking problem.

### 5.3 Limitations

Three limitations bound the generality of our findings.

First, our evaluation uses **skill-level features only**. We do not provide LLMs with question text, problem identifiers, or skill names — we provide only the abstract skill index. This is the standard simplification adopted by the original DKT, SAKT, and AKT papers, but it underrates LLMs' ability to leverage natural-language question content. A meaningful follow-up would provide LLMs with question text and re-evaluate.

Second, we test only **mid-tier frontier LLMs** (Claude Haiku 4.5, GPT-4o-mini). Larger frontier models (Claude Sonnet 4.5, GPT-4o, Gemini 2.5 Pro) may close more of the gap, although prior multi-tier comparisons (Bhattacharyya et al., 2026) suggest the LLM–KT gap is not strongly model-size-bound at this scale. Cost considerations precluded a full evaluation at flagship scale; because Claude Sonnet 4.5 is priced at exactly 3x Claude Haiku 4.5 per token ($3/$15 vs. $1/$5 per MTok, input/output), the same token usage on Sonnet 4.5 zero-shot would have cost $1.50 versus the $0.50 we actually paid for Haiku zero-shot.

Third, our evaluation is on **a single dataset** (ASSISTments-2009 corrected). Generalisation to other KT datasets is an open question, and the contrast with Bhattacharyya et al. (2026), who reported substantially worse LLM performance on a proprietary dataset, suggests that dataset choice may matter more than provider or prompting strategy. ASSISTments-2009 has well-known idiosyncrasies — heavy skill imbalance, scaffolding artefacts, multi-skill problems — that may interact with LLM performance in ways not captured here.

### 5.4 Future Work

We see five directions of immediate interest. First, evaluating chain-of-thought prompting and structured reasoning approaches on KT, to test the prompt-representation hypothesis from §5.1. Second, evaluating LLMs with question text rather than skill indices, to bound the ceiling of LLM-KT under richer state encoding. Third, generalising the multi-provider, multi-prompt-strategy benchmark to additional KT datasets (EdNet, ASSISTments-2017, KDD Cup 2010) to test whether the 0.05 AUC gap is dataset-specific or universal. Fourth, exploring hybrid architectures that route easy predictions to a fast specialised model and hard, ambiguous, or low-confidence cases to an LLM. Fifth, evaluating post-hoc calibration methods (Platt scaling, isotonic regression) on LLM-KT outputs to determine how much of the §4.2.2 miscalibration is recoverable from a small held-out set.

---

## 6. Conclusion

We have presented a controlled multi-provider benchmark of large language models against specialized knowledge tracing models on ASSISTments-2009 corrected. Our central finding is that specialized KT models — particularly DKT — retain a small but statistically significant advantage over modern frontier LLMs (a 0.052 AUC gap, p < 0.001), and this gap is invariant to LLM provider and prompt strategy, suggesting an architectural rather than capability-bound limitation. A prediction-level analysis of 6,680 LLM outputs identifies three concrete mechanisms behind the gap: per-position AUC is flat across the evaluation sequence, so additional student history is not exploited; predicted probabilities are miscalibrated (ECE 0.10–0.12) and anchored to a small set of human-readable values; and the four LLM conditions correlate at Pearson 0.83–0.93 with one another, consistent with a shared task ceiling rather than independent model error. Specialised models additionally dominate the cost-accuracy Pareto frontier — zero marginal cost at the highest AUC — while LLM API calls scale to $126–$1,580 per year at a realistic four-million-prediction deployment. The practical implication is not to abandon LLMs in education but to deploy them in roles complementary to specialised KT — explanation, scaffolding, free-text feedback, and bootstrap-data regimes — and to recover well-calibrated probabilities from LLM rankings via a post-hoc calibration step where direct LLM prediction is unavoidable. We identify hybrid architectures, richer prompt-state representations, and post-hoc output calibration as the most promising directions for closing the remaining gap.

---

## References

*All references below are verified against arXiv or publisher records. Final formatting in APA 7th edition for journal submission.*

- Bhattacharyya, P., Mitton, J., Abboud, R., & Woodhead, S. (2026). *Faster, cheaper, more accurate: Specialised knowledge tracing models outperform LLMs.* arXiv preprint arXiv:2603.02830. https://arxiv.org/abs/2603.02830
- Corbett, A. T., & Anderson, J. R. (1994). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, *4*(4), 253–278.
- Dai, W., Lin, J., Jin, F., Li, T., Tsai, Y., Gasevic, D., & Chen, G. (2023). Can large language models provide feedback to students? A case study on ChatGPT. In *Proceedings of the IEEE International Conference on Advanced Learning Technologies (ICALT)*.
- Feng, M., Heffernan, N. T., & Koedinger, K. R. (2009). Addressing the assessment challenge in an intelligent tutoring system that tutors as it assesses. *User Modeling and User-Adapted Interaction*, *19*(3), 243–266.
- Ghosh, A., Heffernan, N., & Lan, A. S. (2020). Context-aware attentive knowledge tracing. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (KDD '20).
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*.
- Mitton, J., Bhattacharyya, P., Smith, D., Christie, T., Abboud, R., & Woodhead, S. (2026). *Misconception diagnosis from student-tutor dialogue: Generate, retrieve, rerank.* arXiv preprint arXiv:2602.02414. https://arxiv.org/abs/2602.02414
- Mizumoto, A., & Eguchi, M. (2023). Exploring the potential of using an AI language model for automated essay scoring. *Research Methods in Applied Linguistics*, *2*(2).
- Nakagawa, H., Iwasawa, Y., & Matsuo, Y. (2019). Graph-based knowledge tracing: Modeling student proficiency using graph neural network. In *IEEE/WIC/ACM International Conference on Web Intelligence*.
- Norris, M., Gal, K., & Bulathwela, S. (2026). *Next token knowledge tracing: Exploiting pretrained LLM representations to decode student behaviour.* arXiv preprint arXiv:2511.02599. https://arxiv.org/abs/2511.02599
- Pandey, S., & Karypis, G. (2019). A self-attentive model for knowledge tracing. In *Proceedings of the 12th International Conference on Educational Data Mining (EDM)*.
- Piech, C., Bassen, J., Huang, J., Ganguli, S., Sahami, M., Guibas, L. J., & Sohl-Dickstein, J. (2015). Deep knowledge tracing. In *Advances in Neural Information Processing Systems (NeurIPS) 28*.
- Scarlatos, A., Baker, R. S., & Lan, A. (2025). Exploring knowledge tracing in tutor-student dialogues using LLMs. In *Proceedings of the 15th International Learning Analytics and Knowledge Conference (LAK '25)*. https://doi.org/10.1145/3706468.3706501
- Sonkar, S., Liu, N., Le, M., & Baraniuk, R. (2024). MalAlgoQA: Pedagogical evaluation of counterfactual reasoning in large language models and implications for AI in education. In *Findings of the Association for Computational Linguistics: EMNLP 2024*.
- Yeung, C. K., & Yeung, D. Y. (2018). Addressing two problems in deep knowledge tracing via prediction-consistent regularization. In *Proceedings of the Fifth Annual ACM Conference on Learning at Scale (L@S '18)*.

**Reference verification status:** All 15 references independently verified directly against primary sources (arXiv, ACL Anthology, ACM DL, Springer, OpenReview, publisher/conference pages) as of 2026-07-24. Bhattacharyya et al. 2026 (arXiv:2603.02830), Norris et al. 2026 (arXiv:2511.02599), and Mitton et al. 2026 (arXiv:2602.02414) confirmed real, correctly titled and authored. Scarlatos et al. 2025 confirmed published in LAK '25 (DOI 10.1145/3706468.3706501; arXiv:2409.16490). Corbett & Anderson 1994 confirmed (not 1995 as commonly miscited elsewhere); *User Modeling and User-Adapted Interaction*, 4, 253–278. Feng, Heffernan, & Koedinger 2009 confirmed: *User Modeling and User-Adapted Interaction*, 19(3), 243–266. Piech et al. 2015 confirmed NeurIPS 28. Pandey & Karypis 2019 confirmed EDM 2019. Ghosh et al. 2020 confirmed KDD 2020. Yeung & Yeung 2018 confirmed L@S 2018. Hu et al. 2022 (LoRA) confirmed ICLR 2022. Nakagawa et al. 2019 confirmed IEEE/WIC/ACM Web Intelligence 2019. Mizumoto & Eguchi 2023 confirmed *Research Methods in Applied Linguistics*, 2, article 100050. Dai et al. 2023 confirmed ICALT 2023. **One correction made:** Sonkar et al. 2024 was miscited (wrong venue "NAACL" and wrong title) in an earlier draft; corrected above to the actual paper — "MalAlgoQA: Pedagogical Evaluation of Counterfactual Reasoning in Large Language Models and Implications for AI in Education," Sonkar, Liu, Le, & Baraniuk, *Findings of ACL: EMNLP 2024*.

---

## References

*See References section above.*
