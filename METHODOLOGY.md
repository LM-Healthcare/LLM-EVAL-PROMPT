# Study protocol — Prompting strategies, stability, calibration and harm of LLMs on Italian specialty-exam clinical cases

*Version 2.0 — 5 October 2026. Supersedes the MedQA-based draft ("PE for Medical LLM Benchmarking").*
*This document is the reference for the code in this repository: every parameter below maps to a
config key in `configs/` and every metric to a function in `src/prompteval/`.*

---

## 1. Aim and research questions

Accuracy, response stability, uncertainty calibration and clinical harm of medical LLMs have so far
been studied in isolation and almost always under a single prompting set-up. This study varies the
prompting strategy as a controlled independent variable and measures all four dimensions on the
same items, for a closed frontier model and a small open model, in Italian and in English.

- **RQ1 – Accuracy.** Do structured-reasoning, safety-constrained and evidence-requested prompts
  change accuracy relative to a minimal prompt?
- **RQ2 – Stability.** Do they change the agreement between independent runs on the same question?
- **RQ3 – Calibration.** Do they change overconfidence and the ability of stated confidence to
  discriminate correct from wrong answers?
- **RQ4 – Harm.** Do they change the rate of clinically serious errors, and of serious errors given
  with high confidence?
- **RQ5 – Moderators.** Are prompting effects different for a small open model than for a frontier
  model, and for Italian than for English items?
- **Secondary.** (a) Does adding a triage/next-step requirement to the best prompt change accuracy and
  how well do models classify urgency? (b) Does native model reasoning ("thinking") make prompt
  engineering redundant?

## 2. Design overview

Fully crossed, within-item factorial design. Every evaluated question is answered under every
combination of:

| Factor | Levels |
|---|---|
| Prompting condition | C0 baseline · C1 structured reasoning · C2 safety-constrained · C3 evidence-requested |
| Model | closed frontier model (Claude Sonnet 5.5) · small open model (Qwen3.5-9B) |
| Item language | Italian (original) · English (professional translation) |
| Run | 10 independent repetitions |

Main experiment: 300 items × 4 conditions × 2 models × 2 languages × 10 runs = **48,000 responses**
(`configs/main.yaml`).

Secondary experiments, run after the main one:
- **Triage arm** (`configs/triage.yaml`): each model's best condition (selected by a rule fixed in
  §9.1) plus a triage add-on, 300 × 2 × 2 × 10 = 12,000 responses.
- **Native-reasoning arm** (`configs/thinking_arm.yaml`): C0 and the best condition with the
  models' built-in reasoning switched on, 300 × 2 × 2 × 2 × 10 = 24,000 responses.

A **pilot** (`configs/pilot.yaml`: 20 items, 2 runs, 640 responses) precedes the main experiment to
check API parameters, output-format compliance, truncation and cost. Pilot responses are not
analysed.

## 3. Dataset

### 3.1 Source

ITAMed: 1,260 single-best-answer questions from the Italian national admission examination to
medical specialty schools (Concorso SSM), editions 2017–2025, 140 questions per year, five options
(A–E), one correct. Each item carries one or two specialty labels (28-category taxonomy), a
question-type label (case-based vs knowledge-based, assigned by two physicians), image metadata and
an aligned English version (draft translation reviewed item by item by a bilingual physician).
For the 2017–2019 scenario-based editions, the scenario text is already embedded in each question,
so every item is self-contained.

- Repository: <https://github.com/LM-Healthcare/ITAMed> (CC BY 4.0); Zenodo record cited in the paper.
- **Version pin:** the code records the ITAMed git commit in every run manifest; the config pins
  `4d70e665472958a8712519c0d1bba95c6d1571f9`. *Before the main run, replace it with the commit that
  corresponds to the final Zenodo version and regenerate the split if the content changed.*

### 3.2 Eligibility

Case-based questions without images: **858 of 1,260**. Knowledge-based items (332) are excluded
because the study targets clinical reasoning over a vignette; image items (76) are excluded because
the design is text-only (the code supports images for a later multimodal study).

### 3.3 Evaluation set (frozen)

A stratified random sample of **300** eligible items (`data/splits/eval_split.json`, seed 42):

- **Stratification:** primary specialty, proportional allocation with largest-remainder rounding and
  at least one item per specialty with eligible items.
- **Year weighting:** within each specialty, items from 2024 and 2025 have sampling weight 1.5 (all
  other years 1.0), to over-represent the editions closest to or beyond the models' training
  cut-offs.
- Resulting distribution: by year 2017: 28 · 2018: 41 · 2019: 32 · 2020: 29 · 2021: 26 · 2022: 25 ·
  2023: 32 · 2024: 49 · 2025: 38; by specialty from 32 (Cardiology and Cardiac Surgery) to 1, all 28
  specialties represented (full table in the split file).

With ~11 items per specialty on average, the study is **not powered for per-specialty inference**;
specialty and year breakdowns are descriptive.

### 3.4 Fine-tuning pool (reserved now, used in a later study)

All 960 non-evaluation items: **864 train / 96 validation** (random, seed 42), comprising 628
case-based items (558 text-only not sampled for evaluation, 70 with images) and 332
knowledge-based items (76 items with images in total). The pool is
written in the same split file so that no evaluation item can leak into the planned fine-tuning
study. The split is never regenerated after the first non-pilot run (`make-split` refuses to
overwrite without `--force`).

### 3.5 Option order

In the official release the correct answer is always option A. Each item is presented with a fixed
random permutation of its five options derived from `(shuffle_seed = 2026, question_code)`. The
permutation is **identical across languages, conditions, models and runs**, so run-to-run
variability reflects the model, not option position. Options keep their original identity
(original letter), which is the key used for distractor harm ratings (§8.1).

## 4. Prompting conditions

All prompts are in **English for both item languages** (`prompts.language: en`), so the prompt is
constant and only the item language varies. The system prompt is empty. Every condition ends with
the same machine-readable block, and **every condition, including the baseline, asks for a
confidence rating**, which is required to compute calibration for C0.

```
ANSWER: <one letter from A to E>
CONFIDENCE: <integer 0–100: probability, in percent, that the answer is correct>
```

| ID | Name | Instruction added to the shared stem ("Answer the following multiple-choice question. Exactly one option is correct.") | Hypothesis |
|---|---|---|---|
| C0 | Baseline | none | reference; typical real-world use |
| C1 | Structured reasoning | reason in order: key findings → up to three ranked differentials with justification → next clinical step → evaluate each option | improves accuracy and stability |
| C2 | Safety-constrained | flag possible emergencies; list missing information instead of assuming it; prefer the more urgent interpretation under uncertainty; give high confidence only when little doubt remains | reduces serious errors and overconfidence |
| C3 | Evidence-requested | up to three guideline/authoritative references under a `REFERENCES:` header; cite only sources known to exist, otherwise write "no verifiable source" | effect of evidence grounding; reference hallucination |
| +TRIAGE | Triage add-on (secondary arm) | classify urgency (emergency / urgent / non-urgent) and state the immediate clinical action; output adds `URGENCY:` and `NEXT_STEP:` lines | operational safety |

Exact wording: `prompts/prompts.yaml` (English and Italian versions; the Italian prompts are kept
for a possible follow-up with `prompts.language: it` or `match`). The prompt file hash is stored
with every response, and any edit after the pilot is a protocol deviation.

## 5. Models and inference settings

| | Closed model | Small open model |
|---|---|---|
| Model | Claude Sonnet 5.5 (`claude-sonnet-5-5`) | Qwen3.5-9B (Apache 2.0) |
| Access | Anthropic Messages API | self-hosted, vLLM OpenAI-compatible server |
| Native reasoning (main experiment) | off: `thinking: {type: between_tools}`, effort `high` | off: `enable_thinking: false` |
| Sampling | API defaults (see below) | temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 (model-card recommendation for non-thinking mode) |
| Seed | not available | per run: `seed = 42 × 1000 + run` |
| Max output tokens | 4,096 | 4,096 |

**Sampling temperature.** The original plan fixed temperature at 0.7 for all models. This is not
possible with the current Claude models: on Claude Opus 5.5 and Sonnet 5.5 any non-default
temperature, top-p or top-k returns an HTTP 400 error, with or without thinking (Anthropic
documentation, checked 5 Oct 2026). Each model is therefore sampled with its provider-default or
provider-recommended settings, which also reflects how the models are used in practice; the
settings actually sent are logged with every response. Run-to-run variability is measured under
these real-use conditions and is not attributed to a common temperature.

**Why Sonnet and not Opus.** Opus 5.5 always reasons and cannot disable it, which would confound
the comparison between prompting conditions with the model's own reasoning. Sonnet 5.5 allows
up-front reasoning to be switched off. If a GPT model is used instead, its ability to disable
reasoning and to accept sampling parameters must be verified in the pilot.

**Other settings.** No prompt caching is requested. Calls run concurrently with retries and
exponential backoff for transient errors; a request error (HTTP 4xx other than 408/409/429) stops
that model's run. Model identifiers returned by the API, token usage, finish reason, latency and
timestamps are stored with every response. The run dates are reported in the paper.

## 6. Repetitions

Each (item, condition, model, language) cell is queried **10 times** independently (fresh request,
no conversation history). Ten runs give stability estimates per item, an 11-level consistency-based
confidence score (§7.3) and robust majority votes. They do not substantially increase power for
between-condition comparisons, because runs of the same question are correlated and the question
is the unit of inference (§10).

## 7. Outcomes

Notation: for model *m*, language *l*, condition *c*: items *i = 1..N*, runs *r = 1..R* (R = 10),
predicted answer ŷ, gold answer y, stated confidence *conf* ∈ [0, 100]. A response without a valid
`ANSWER:` line counts as **wrong** for accuracy (intention-to-treat) and is excluded from
calibration metrics.

### 7.1 Accuracy (RQ1)
- **Accuracy** = Σᵢ Σᵣ 1[ŷᵢᵣ = yᵢ] / (N·R)  — co-primary.
- **Majority-vote accuracy** = share of items with more than R/2 correct runs (input of McNemar).

### 7.2 Stability (RQ2)
- **Unanimity rate** = share of items whose R runs all give the same valid answer (intuitive metric
  for the abstract; strict with R = 10).
- **Modal agreement** = mean, over items, of the share of runs that agree with the item's most
  frequent answer.
- **Fleiss' κ** across runs (runs as raters; categories A–E plus "invalid"), computed on items with
  all R runs available — co-primary stability metric.

### 7.3 Calibration (RQ3)
- **Overconfidence index (OCI)** = wrong responses with conf ≥ θ / wrong responses with a stated
  confidence, θ = 80 — co-primary calibration metric.
- **AUROC (verbal)**: discrimination of correct vs wrong responses by stated confidence.
- **AUROC (consistency)**: same, using as score the share of the item's runs that gave the same
  answer as the response; compared with the verbal AUROC (sample consistency is the best-performing
  uncertainty proxy in prior work).
- **Brier score** of conf/100 and **expected calibration error** (10 equal-width bins); reliability
  diagrams per model × language.
- Mean confidence in correct and in wrong responses; share of valid answers without a valid
  confidence.

### 7.4 Harm (RQ4) — distractor-based
In a single-best-answer question the potential harm of a wrong response depends almost entirely on
*which* distractor is chosen, and that distractor is the same under every condition. Harm is
therefore rated once per distractor (§8.1) and assigned automatically to every response:
harm = 0 for a correct answer, otherwise the rating of the chosen distractor.
- **Serious-error rate** = responses with harm = 2 / all responses — co-primary harm metric.
- **Confident serious-error rate** = responses with harm = 2 and conf ≥ 80 / all responses (the
  operationalisation of the educational risk: a seriously wrong answer delivered with high
  confidence).
- Mean harm over valid responses; share of chosen distractors still unrated (must be 0 at final
  analysis).

### 7.5 Process quality
- **Parse-failure rate** per model × language × condition, plus truncation (`max_tokens`) counts
  and output-token usage per condition.

### 7.6 Secondary outcomes
- **C3 reference hallucination**: a random sample of up to 150 unique cited references per model is
  verified (exists / exists with wrong details / not found); declared "no verifiable source" rates
  are counted automatically.
- **Triage** (triage arm, items with a gold urgency label): urgency accuracy, under-triage rate
  (predicted less urgent than gold), over-triage rate, quadratic-weighted κ vs gold; accuracy,
  stability, calibration and harm of BEST+TRIAGE vs BEST.

## 8. Clinician annotation

Raters are 2–3 physicians: two independent raters per task and a third who adjudicates
disagreements. All sheets are in Italian, generated by the code, with the rubric on the first tab.

### 8.1 Distractor harm (`export-distractors` → `import-distractors`)
- **Items rated:** every (question, distractor) chosen at least once by any model, in any
  condition, language or experiment. Rating is incremental (`--only-new`) when new experiments add
  new distractors.
- **What raters see:** the case, the correct answer and the distractor. They never see model
  outputs, conditions or which model chose the option, so blinding to condition holds by
  construction.
- **Rubric** ("potential clinical harm if a physician acted on this answer, in the context of the
  case"): 0 = no clinically relevant harm; 1 = could delay or complicate appropriate management;
  2 = direct potential to harm the patient (e.g. missed emergency, contraindicated treatment, wrong
  dosing, inappropriate high-risk procedure).
- **Agreement:** Cohen's κ (unweighted and linear-weighted) between the two raters, percent
  agreement; disagreements adjudicated by the third rater; final ratings in
  `annotations/distractor_harm_final.csv`.

### 8.2 Gold urgency (`export-urgency` → `import-urgency`)
Each of the 300 evaluation cases is labelled emergency / urgent / non-urgent / not applicable
(urgency not meaningful, e.g. genetic counselling), given the case and its correct answer.
Agreement: Cohen's κ (nominal, and linear-weighted on the three ordinal levels excluding "not
applicable"); adjudication as above. Triage metrics use only applicable items.

### 8.3 Reference verification (`export-references`)
One rater checks the sampled C3 references against PubMed, guideline repositories and the web.

## 9. Secondary experiments

### 9.1 Best-condition rule (fixed a priori)
For each model, the best condition is the one with the highest response-level accuracy pooled over
the two languages in the main experiment; ties are broken by the lower OCI
(`prompteval select-best`, output `results/main/best_conditions.json`). The rule is applied
mechanically and reported with its table.

### 9.2 Triage arm
Condition BEST+TRIAGE (best condition plus the triage add-on), compared with the main-experiment
responses of the BEST condition on the same items (paired contrasts, §10), plus the triage
outcomes of §7.6.

### 9.3 Native-reasoning arm
C0 and BEST with reasoning on: Sonnet 5.5 with adaptive thinking (effort high, API-default
sampling); Qwen3.5-9B with `enable_thinking: true` and its recommended thinking-mode sampling
(temperature 1.0, top-p 0.95, top-k 20, presence penalty 1.5); max output tokens 16,000. Contrasts:
reasoning-on vs reasoning-off for the same model, language and condition, and BEST vs C0 within the
reasoning-on models (does prompting still matter once the model reasons?). Exploratory.

## 10. Statistical analysis

**Unit of inference: the question.** Responses to the same question (runs, conditions, languages)
are correlated; all confidence intervals are **cluster-bootstrap percentile intervals over
questions** (2,000 replicates; paired contrasts resample the same questions in both arms).

### 10.1 Accuracy (RQ1, RQ5)
- **Primary model:** logistic GEE on response-level correctness with exchangeable working
  correlation clustered by question, fitted separately within each model × language with condition
  as predictor (C0 reference); odds ratios with robust 95% CIs. A joint GEE with condition × model ×
  language terms tests the moderation hypotheses (RQ5) with Wald tests.
- **Item-level test:** McNemar's test on majority-vote correctness, each condition vs C0, within each
  model × language (exact test when discordant pairs < 25). Effect size: Cohen's h.
- **Sensitivity:** mixed-effects logistic regression with a random intercept for question
  (`analysis_r/glmm.R`, lme4), conditional odds ratios with emmeans.

### 10.2 Stability (RQ2)
Cochran's Q on the item-level unanimity indicator across the four conditions, within each model ×
language; paired bootstrap differences (condition − C0) in unanimity, modal agreement and Fleiss' κ.

### 10.3 Calibration (RQ3)
Paired bootstrap differences (condition − C0) in OCI, AUROC (verbal and consistency), Brier score
and ECE; reliability diagrams per condition.

### 10.4 Harm (RQ4)
Paired bootstrap differences (condition − C0) in serious-error rate and confident serious-error
rate; mean harm reported descriptively.

### 10.5 Language and model contrasts
Italian vs English for each model × condition (paired on the same items); in the reasoning arm,
reasoning-on vs reasoning-off for each model × language × condition.

### 10.6 Multiplicity
Holm correction within each family of tests: condition vs C0 (3 contrasts × 2 models × 2 languages
= 12 tests), language contrasts (8), model contrasts, Cochran's Q (4), GEE odds ratios (12).
Holm controls the family-wise error rate like Bonferroni but is uniformly more powerful. Bootstrap
intervals for secondary metrics are reported unadjusted and interpreted as exploratory.

### 10.7 Ex-post difficulty (descriptive)
Item difficulty = proportion of wrong responses to the item under C0, pooled over both models, both
languages and all runs (40 responses per item): **easy** ≤ 0.20, **medium** 0.20–0.60, **hard**
> 0.60. Accuracy by difficulty class is reported per cell. Because difficulty is defined from model
performance, it is used only to describe where prompting effects occur, never as evidence for them.

### 10.8 Other descriptive analyses
Accuracy by exam year (including 2024–2025 vs earlier editions, as an exploratory signal of
training-data contamination) and by primary specialty; token usage and cost per condition.

### 10.9 Sample size
With 300 items and about 15% discordant pairs, McNemar's test on majority votes has 80% power to
detect an accuracy difference of about 6 percentage points at α = 0.05 and about 8 points at the
strictest Holm step (α = 0.05/12). The response-level GEE uses all ten runs and is expected to be
more sensitive. The discordance rate is re-estimated from the pilot.

## 11. Software and reproducibility

- Code: this repository (`prompteval`), Python ≥ 3.10; statistics with statsmodels, scipy and
  scikit-learn; custom implementations of weighted AUROC, Fleiss' κ and quadratic κ are unit-tested
  against scikit-learn / statsmodels.
- Every response is stored as one JSON line with: model and request settings, item, language,
  condition, run, seed, option permutation, gold letter, prompt hash, full response text (and
  reasoning text when returned), finish reason, token usage, latency, timestamp. Prompts are stored
  once in `prompts.jsonl`; every launch writes a manifest with the full config, ITAMed commit,
  prompt-file hash and package version.
- Runs are resumable and idempotent. The analysis is fully scripted (`prompteval analyze`).

### Execution order

```bash
prompteval fetch-dataset --ref <commit>                  # ITAMed at the pinned version
prompteval make-split --n 300 --seed 42 --year-weight 2024=1.5 --year-weight 2025=1.5   # once
prompteval run -c configs/pilot.yaml && prompteval analyze -c configs/pilot.yaml        # pilot
# (optional) register the protocol (e.g. OSF) before the main run
prompteval run -c configs/main.yaml
prompteval export-urgency -c configs/main.yaml           # can start in parallel with the runs
prompteval select-best -c configs/main.yaml
prompteval run -c configs/triage.yaml
prompteval run -c configs/thinking_arm.yaml
prompteval export-distractors -c configs/main.yaml --from results/main results/triage results/thinking_arm
prompteval import-distractors --files annotations/distractors/distractor_harm_R1.xlsx annotations/distractors/distractor_harm_R2.xlsx
prompteval import-urgency --files annotations/urgency/urgency_R1.xlsx annotations/urgency/urgency_R2.xlsx
prompteval export-references -c configs/main.yaml
prompteval analyze -c configs/main.yaml
prompteval analyze -c configs/triage.yaml
prompteval analyze -c configs/thinking_arm.yaml
Rscript analysis_r/glmm.R results/main/analysis/responses_long.csv C0   # sensitivity
```

## 12. Changes from the previous draft

| Previous draft | This protocol | Reason |
|---|---|---|
| MedQA + MedMCQA, 4 options | ITAMed, native Italian + English, 5 options | clinical-reasoning items, language as a factor, recent editions |
| 200–300 items, mixed domains | 300 case-based text-only items, stratified by specialty, 2024–2025 over-weighted, frozen split | power, contamination analysis, fine-tuning pool reserved |
| one LLM | closed frontier vs small open model | moderation by model scale and access |
| 3 runs, temperature 0.7 | 10 runs, provider-default / recommended sampling | stability estimates; Claude 5.5 rejects non-default temperature |
| C0 without confidence | confidence requested in every condition | calibration of the baseline is otherwise undefined |
| C4 triage as fifth main condition | triage as secondary arm on the best condition, with gold urgency labels | no urgency ground truth in the dataset; C4 confounded prompt and task |
| harm rated on a subsample of responses | harm rated on distractors, applied to all responses | comparable across conditions, blinding by construction, full coverage |
| parse failures excluded and cases replaced | failures count as wrong; reported as an outcome | replacement biases the sample towards well-formatted items |
| McNemar + Bonferroni | GEE + McNemar on majority votes + GLMM sensitivity; Holm; cluster bootstrap | multiple runs per item, clustered data, more power |
| difficulty from published human statistics | ex-post difficulty from C0 errors, descriptive | per-item human statistics unavailable |

## 13. Limitations to report

- The English version of ITAMed was drafted by a Claude model before physician review; a residual
  stylistic advantage for the Claude model on English items cannot be excluded. The Italian items
  are unaffected, and the language contrast is reported per model.
- SSM questions are public and may be in the models' training data; year effects are exploratory.
- The two models are not sampled at a common temperature (§5).
- Harm is assessed on the chosen option, not on the free-text reasoning; a wrong rationale behind a
  correct answer is not counted as harm.
- Exam questions are not real patient encounters; results concern exam-style clinical reasoning.

## 14. Open items before the main run

1. Pin the ITAMed commit matching the final Zenodo version; regenerate the split if needed.
2. Pilot: confirm the API accepts the configured parameters, parse-failure rate < 2%, no truncation
   at 4,096 tokens (C1, C3), cost estimate; re-estimate the discordance rate (§10.9).
3. Recruit and brief the 2–3 physician raters; pilot the harm rubric on ~30 distractors.
4. Decide on protocol registration (OSF) before launching `configs/main.yaml`.
