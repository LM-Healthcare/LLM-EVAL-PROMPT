# LLM-EVAL-PROMPT

Research project evaluating Large Language Model (LLM) performance on medical multiple-choice questions using different prompting strategies.

## Dataset

We use the [MedMCQA](https://medmcqa.github.io/) dataset, which contains ~193k medical MCQs across 20 clinical subjects. Questions are sourced from Indian medical entrance exams (AIIMS/PGI).

### Subject Split

The original dataset (dev / test / train) has been split into **20 per-subject JSON files** for stratified sampling. After filtering out questions with unknown subjects, **189,426 valid questions** remain.

See [`MedMCQA/report.md`](MedMCQA/report.md) for a detailed breakdown by subject, split, choice type, and explanation availability.

## Project Structure

```
├── .gitignore
├── README.md
├── Metodologia.txt              # Methodology notes (Italian)
├── MedMCQA/
│   ├── split_by_subject.py      # Script to split dataset by subject
│   ├── report.md                # Per-subject statistics report
│   └── data/                    # (gitignored) original + split JSON files
└── Articoli rilevanti/          # (gitignored) reference papers
```

## Methodology (Summary)

- **200 cases** sampled from MedMCQA via stratified sampling across 4 macro-categories:
  - Emergency Medicine (50)
  - Internal Medicine (50)
  - Pharmacology (50)
  - Mixed (50)
- **5 independent runs** per case
- **Temperature 0.7** across all runs
- Domain classification performed automatically via LLM, with manual validation on a 10% sample

## Status

- [x] Dataset acquisition and per-subject splitting
- [ ] Stratified case sampling (200 cases across 4 macro-categories)
- [ ] LLM evaluation runs
- [ ] Analysis and paper writing
