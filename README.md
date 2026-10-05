# LLM-EVAL-PROMPT

Controlled evaluation of **prompting strategies** for medical LLMs on **ITAMed** (Italian specialty
admission exam, 2017–2025, Italian + English). For every question, condition, model, language and
run it measures **accuracy, run-to-run stability, confidence calibration and clinical harm**, and
it supports the clinician-annotation workflows the study needs.

The study protocol is in **[METHODOLOGY.md](METHODOLOGY.md)**. This README covers how to use the code.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
prompteval fetch-dataset --dest data/ITAMed --ref <ITAMed commit>
```

API keys are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or the
variable named by `api_key_env` in a model entry). The open model is served locally, e.g.
`vllm serve Qwen/Qwen3.5-9B --max-model-len 16384`.

## Workflow

```bash
prompteval make-split --n 300 --seed 42 --year-weight 2024=1.5 --year-weight 2025=1.5  # once, then frozen
prompteval dry-run  -c configs/pilot.yaml      # task counts + rendered prompts, no API calls
prompteval run      -c configs/pilot.yaml      # 20 items x 2 runs: check parameters, format, cost
prompteval analyze  -c configs/pilot.yaml
prompteval run      -c configs/main.yaml       # resumable: re-run the same command after an interruption
prompteval status   -c configs/main.yaml
prompteval select-best -c configs/main.yaml    # a-priori rule -> results/main/best_conditions.json
prompteval run      -c configs/triage.yaml
prompteval run      -c configs/thinking_arm.yaml
prompteval analyze  -c configs/main.yaml       # also triage.yaml, thinking_arm.yaml
```

Useful options: `run --model sonnet55` (one model only), `run --max-tasks 100` (stop after 100 new
calls per model).

### Clinician annotations

```bash
# distractor harm: sheets for every distractor chosen by any model (blind to model and condition)
prompteval export-distractors -c configs/main.yaml --from results/main results/triage results/thinking_arm --raters R1 R2
prompteval import-distractors --files annotations/distractors/distractor_harm_R1.xlsx annotations/distractors/distractor_harm_R2.xlsx
#   -> writes an adjudication sheet for disagreements; fill it and pass it with --adjudication
#   -> later experiments: export-distractors --only-new

# gold urgency for the triage arm
prompteval export-urgency -c configs/main.yaml --raters R1 R2
prompteval import-urgency --files annotations/urgency/urgency_R1.xlsx annotations/urgency/urgency_R2.xlsx

# C3 references to verify
prompteval export-references -c configs/main.yaml --per-model 150
```

`analyze` picks up `annotations/distractor_harm_final.csv` and `annotations/urgency_final.csv`
automatically (paths in `analysis.annotations`).

## Configuration

Experiments are YAML files in `configs/`; a config can `extends:` another and override only what
changes (dicts are merged, lists replaced). Main keys:

| Key | Meaning |
|---|---|
| `dataset.languages` | `[it]`, `[en]` or `[it, en]` (item language) |
| `dataset.split_file` | frozen evaluation set; without it, items come from `dataset.filters` |
| `dataset.filters` | `years` (list or `{min, max}`), `question_types`, `include_images`, `specialties`, `question_codes` |
| `dataset.limit` | keep the first N items (pilots, smoke tests) |
| `prompts.language` | `en`, `it` or `match` (prompt in the item's language) |
| `prompts.conditions` | e.g. `[C0, C1, C2, C3]`; add-ons with `+` (`C1+TRIAGE`); `BEST` alias after `select-best` |
| `sampling.runs` | repetitions per item |
| `models` | one entry per model: `id`, `backend` (`anthropic`, `openai`, `openai_compatible`, `mock`), `model`, `max_tokens`, `params` (top-level request fields), `extra_body` (sent verbatim), `seed_per_run`, `concurrency`, `requests_per_minute` |
| `analysis` | reference condition, OCI threshold, bootstrap replicates, contrasts, annotation files, imported experiments |

New prompting conditions or add-ons are added in `prompts/prompts.yaml` (no code change). New API
fields go in `params` / `extra_body` (no code change).

## Outputs

```
results/<experiment>/
  responses/<model_id>.jsonl   one line per response (full text, settings, seed, permutation, usage)
  prompts.jsonl                every distinct prompt, by hash
  manifests/                   config, ITAMed commit, prompt-file hash per launch
  errors.jsonl                 calls that failed after all retries (re-run to retry them)
  analysis/
    report.md                  summary tables
    metrics_by_cell.csv        all metrics with 95% cluster-bootstrap CIs
    contrasts.csv              condition vs reference, IT vs EN, model pairs (McNemar + Holm, paired bootstrap)
    gee_condition_effects.csv  odds ratios vs reference; gee_interactions.txt (joint model)
    cochran_q_unanimity.csv, accuracy_by_{difficulty,year,specialty}.csv, reliability_bins.csv, usage.csv
    reliability_*.png, accuracy_by_condition.png
    responses_long.csv         input for analysis_r/glmm.R (mixed-effects sensitivity analysis)
```

## Tests

```bash
pytest                                              # unit + end-to-end tests on a synthetic dataset
prompteval run -c configs/smoke_mock.yaml           # full pipeline on real ITAMed items with simulated models
prompteval analyze -c configs/smoke_mock.yaml
```

## Repository layout

```
configs/        main, pilot, triage, thinking_arm, smoke_mock
prompts/        prompts.yaml (EN + IT)
data/splits/    eval_split.json (frozen evaluation set + fine-tuning pool)
src/prompteval/ dataset, split, prompts, backends/, runner, parsing, metrics, stats, analysis, annotations, plots, cli
analysis_r/     glmm.R
tests/
```
