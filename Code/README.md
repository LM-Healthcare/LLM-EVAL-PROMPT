# Classification Pipeline — Technical Documentation

## Overview

This pipeline classifies medical multiple-choice questions from the [MedMCQA](https://medmcqa.github.io/) dataset into **4 macro-categories** using two independent LLMs, then compares their outputs to select a final set of 200 high-quality, consistently-classified questions.

### Macro-Categories

| # | Category | Description |
|--:|----------|-------------|
| 1 | **Emergency Medicine** | Clinical vignettes involving acute, life-threatening conditions typically seen in emergency departments (e.g. bacterial meningitis, myocardial infarction, sepsis, pulmonary embolism, acute trauma, anaphylaxis). |
| 2 | **Internal Medicine** | Clinical reasoning about chronic or systemic diseases with a subacute course (e.g. autoimmune diseases, chronic kidney disease, decompensated diabetes, COPD management, thyroid disorders). |
| 3 | **Pharmacology** | Questions focused on mechanisms of action, side effects, drug interactions, contraindications, dosages, and pharmacokinetics/pharmacodynamics. |
| 4 | **Mixed** | Questions spanning multiple specialties or not fitting the above categories, including pediatrics, surgery (non-emergency), neurology, obstetrics, psychiatry, basic sciences, etc. |

---

## Pipeline Steps

```
MedMCQA JSONL files
       │
       ▼
[1] split_by_subject.py     →  20 per-subject .jsonl files
       │
       ▼
[2] sample_candidates.py    →  sampled_candidates.jsonl  (~1500 questions)
       │
       ▼
[3] classify_candidates.py  →  classified_model_a.jsonl  (Model A results)
    classify_candidates.py  →  classified_model_b.jsonl  (Model B results)
       │
       ▼
[4] compare_and_select.py   →  final_200_questions.jsonl + report
```

### Step 1 — Split by Subject (`split_by_subject.py`)

Reads the original MedMCQA files (`dev.json`, `test.json`, `train.json` — all in JSONL format), filters out questions with invalid/unknown `subject_name`, and writes one `.jsonl` file per medical subject (20 subjects total). Each line includes a `"split"` field indicating its origin.

### Step 2 — Sample Candidates (`sample_candidates.py`)

Loads the per-subject JSONL files, keeps only `dev` and `train` splits (test split has no correct answer), and samples ~1500 questions using **proportional allocation with a minimum floor**:

- Each subject gets at least `MIN_PER_SUBJECT` (30) questions.
- Remaining slots are distributed proportionally to each subject's size.
- Random seed is fixed (`RANDOM_SEED = 42`) for reproducibility.

Output: `output/sampled_candidates.jsonl`

### Step 3 — LLM Classification (`classify_candidates.py`)

This is the core step. Each sampled question is sent to an LLM with a structured prompt, and the model returns a JSON classification.

#### How It Works

1. **System prompt** defines the 4 categories with detailed descriptions and instructs the model to output structured JSON.

2. **User prompt** provides the question text, all 4 answer options, and the original subject name as context.

3. **LLM response** must be a single JSON object:
   ```json
   {
     "category": "Emergency Medicine",
     "confidence": "high",
     "reasoning": "This question describes an acute MI presentation in the ER."
   }
   ```

#### Confidence Level — How It Is Determined

**The confidence level is entirely self-reported by the LLM.** It is NOT a statistical probability or a token-level logprob. The model is asked to introspect on its own certainty and report one of three levels:

| Level | Meaning |
|-------|---------|
| **high** | The model considers the classification unambiguous — the question clearly fits one category based on the clinical scenario, pathology type, and answer options. |
| **medium** | The question could plausibly belong to two categories, but one is more appropriate. For example, a pharmacology question embedded in an emergency scenario. |
| **low** | The model is uncertain — the question crosses multiple categories or lacks enough clinical context to decide confidently. |

**Important caveats:**
- This is a *qualitative self-assessment*, not a calibrated probability. LLMs tend to be overconfident, so most responses will be "high".
- `temperature=0.0` is used to ensure deterministic outputs — running the same question twice should produce the same classification.
- The confidence level is used downstream (in Step 4) as a **selection signal**: questions classified with "high" confidence by both models are preferred over "low" confidence ones.

#### Response Parsing

The raw LLM output is parsed with `parse_llm_response()`:

1. Strips markdown code fences (` ```json ... ``` `) if present.
2. Parses the JSON object.
3. Validates the `category` field against the 4 allowed values.
4. If the category doesn't match exactly, tries a case-insensitive fuzzy match.
5. If parsing fails entirely, falls back to `{"category": "Mixed", "confidence": "low", "reasoning": "[PARSE ERROR]..."}`.

#### Error Handling and Resume

- **Retries**: Each question is retried up to `MAX_RETRIES` (3) times with exponential backoff (`RETRY_DELAY` × 2^attempt).
- **Concurrency**: Up to `MAX_CONCURRENT_REQUESTS` (10) questions are classified in parallel.
- **Timeout**: Each API call has a `REQUEST_TIMEOUT` (30s) limit.
- **Incremental saves**: Results are saved in **batches of 50** by appending to the output JSONL file. If the script crashes (e.g. quota exhaustion), you can simply re-run the same command and it will **resume from where it stopped** — it counts existing lines in the output file and skips that many candidates.

#### Models Used

| Key | Provider | Model | API Key Env Var |
|-----|----------|-------|-----------------|
| `model_a` | Anthropic | `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| `model_b` | OpenAI | `gpt-5.4` | `OPENAI_API_KEY` |

Both models classify the exact same 1500+ questions independently with the same prompt.

### Step 4 — Compare and Select (`compare_and_select.py`)

Loads both classified files, matches questions by `id`, and computes:

1. **Inter-model agreement rate**: % of questions where both models assigned the same category.
2. **Per-category agreement**: breakdown by macro-category.

Then selects the final 200 questions (50 per category) using a priority system:

| Priority | Criteria |
|----------|----------|
| 1 (highest) | Both models agree + both report "high" confidence |
| 2 | Both models agree + at least one reports "medium" confidence |
| 3 | Both models agree + any confidence |
| 4 (lowest) | Models disagree — model_a's category is used, flagged for manual review |

Within each priority tier, questions are sampled to maximize **subject diversity** (spread across as many different medical subjects as possible).

Output files:
- `final_200_questions.jsonl` — all 200 selected questions
- `final_emergency_medicine.jsonl`, `final_internal_medicine.jsonl`, `final_pharmacology.jsonl`, `final_mixed.jsonl` — 50 each
- `full_comparison.jsonl` — audit trail of all comparisons
- `comparison_report.md` — human-readable summary with statistics

---

## Configuration

All parameters are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RANDOM_SEED` | 42 | Seed for reproducible sampling |
| `TOTAL_CANDIDATES` | 1500 | Target number of questions to sample |
| `MIN_PER_SUBJECT` | 30 | Minimum questions per medical subject |
| `MAX_CONCURRENT_REQUESTS` | 10 | Parallel API calls |
| `REQUEST_TIMEOUT` | 30s | Per-call timeout |
| `MAX_RETRIES` | 3 | Retry count on API errors |
| `RETRY_DELAY` | 2s | Base delay (doubles each retry) |

---

## Setup and Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Create `.env` file

```bash
cp .env.example .env
# Edit .env and add your real API keys
```

The `.env` file is gitignored and loaded automatically via `python-dotenv`.

### 3. Run the pipeline

```bash
# Step 1: Split dataset by subject (run from MedMCQA/ folder)
python split_by_subject.py

# Step 2: Sample candidates (run from Code/ folder)
python sample_candidates.py

# Step 3: Classify with both models
python classify_candidates.py --model model_a
python classify_candidates.py --model model_b

# Step 4: Compare and select final 200
python compare_and_select.py
```

If a classification run is interrupted, simply re-run the same command — it will resume automatically.

---

## File Formats

All data files use **JSONL** (JSON Lines) format: one JSON object per line. This makes it easy to:
- Count questions: `(Get-Content file.jsonl).Count` (PowerShell)
- Stream/process large files without loading everything into memory
- Append results incrementally

Each classified question has this structure:
```json
{
  "id": "abc123",
  "question": "A 45-year-old male presents with...",
  "opa": "Option A", "opb": "Option B", "opc": "Option C", "opd": "Option D",
  "cop": 1,
  "subject_name": "Medicine",
  "topic_name": "Cardiology",
  "split": "train",
  "classification": {
    "category": "Internal Medicine",
    "confidence": "high",
    "reasoning": "Chronic heart failure management question with subacute presentation."
  }
}
```
