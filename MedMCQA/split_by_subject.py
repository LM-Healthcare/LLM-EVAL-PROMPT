"""
Script to split MedMCQA dataset by subject_name.
- Reads dev.json, test.json, train.json (JSONL format)
- Filters out questions with subject_name = null, "Unknown", or not in VALID_SUBJECTS
- Creates one JSON file per subject in the output folder
- Generates a markdown report with statistics
"""

import json
import os
from collections import defaultdict, Counter

# --- Configuration ---

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "by_subject")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "report.md")

SOURCE_FILES = ["dev.json", "test.json", "train.json"]

# Valid subject names as they appear in the dataset
VALID_SUBJECTS = {
    "Anaesthesia",
    "Anatomy",
    "Biochemistry",
    "Dental",
    "ENT",
    "Forensic Medicine",
    "Gynaecology & Obstetrics",
    "Medicine",
    "Microbiology",
    "Ophthalmology",
    "Orthopaedics",
    "Pathology",
    "Pediatrics",
    "Pharmacology",
    "Physiology",
    "Psychiatry",
    "Radiology",
    "Skin",
    "Social & Preventive Medicine",
    "Surgery",
}

# Friendly display names for the report
DISPLAY_NAMES = {
    "Anaesthesia": "Anesthesia",
    "Anatomy": "Anatomy",
    "Biochemistry": "Biochemistry",
    "Dental": "Dental",
    "ENT": "ENT",
    "Forensic Medicine": "Forensic Medicine (FM)",
    "Gynaecology & Obstetrics": "Obstetrics and Gynecology (O&G)",
    "Medicine": "Medicine",
    "Microbiology": "Microbiology",
    "Ophthalmology": "Ophthalmology",
    "Orthopaedics": "Orthopedics",
    "Pathology": "Pathology",
    "Pediatrics": "Pediatrics",
    "Pharmacology": "Pharmacology",
    "Physiology": "Physiology",
    "Psychiatry": "Psychiatry",
    "Radiology": "Radiology",
    "Skin": "Skin",
    "Social & Preventive Medicine": "Preventive & Social Medicine (PSM)",
    "Surgery": "Surgery",
}

# Safe filename mapping
FILE_NAMES = {
    "Anaesthesia": "Anaesthesia",
    "Anatomy": "Anatomy",
    "Biochemistry": "Biochemistry",
    "Dental": "Dental",
    "ENT": "ENT",
    "Forensic Medicine": "Forensic_Medicine",
    "Gynaecology & Obstetrics": "Gynaecology_Obstetrics",
    "Medicine": "Medicine",
    "Microbiology": "Microbiology",
    "Ophthalmology": "Ophthalmology",
    "Orthopaedics": "Orthopaedics",
    "Pathology": "Pathology",
    "Pediatrics": "Pediatrics",
    "Pharmacology": "Pharmacology",
    "Physiology": "Physiology",
    "Psychiatry": "Psychiatry",
    "Radiology": "Radiology",
    "Skin": "Skin",
    "Social & Preventive Medicine": "Social_Preventive_Medicine",
    "Surgery": "Surgery",
}


def load_jsonl(filepath):
    """Load a JSONL file and return a list of dicts."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Data structures for aggregation
    # subject -> split -> [questions]
    subject_questions = defaultdict(lambda: defaultdict(list))
    # Track filtered-out questions
    filtered_out = defaultdict(list)
    # Per-split totals
    split_totals = Counter()
    # Per-subject choice_type counts
    subject_choice_types = defaultdict(Counter)
    # Per-subject has_explanation counts
    subject_has_exp = defaultdict(Counter)

    print("Loading data...")
    for src_file in SOURCE_FILES:
        split_name = src_file.replace(".json", "")  # dev, test, train
        filepath = os.path.join(DATA_DIR, src_file)
        items = load_jsonl(filepath)
        split_totals[split_name] = len(items)
        print(f"  {src_file}: {len(items)} questions")

        for item in items:
            subj = item.get("subject_name")
            # Filter out null, None, empty, "Unknown", or anything not in VALID_SUBJECTS
            if not subj or subj.strip() == "" or subj not in VALID_SUBJECTS:
                filtered_out[split_name].append(item)
                continue
            subject_questions[subj][split_name].append(item)
            # Track choice_type
            ct = item.get("choice_type", "unknown")
            subject_choice_types[subj][ct] += 1
            # Track explanation availability
            exp = item.get("exp")
            has_exp = exp is not None and str(exp).strip() != "" and str(exp).lower() != "null"
            subject_has_exp[subj]["with_exp" if has_exp else "without_exp"] += 1

    # --- Write per-subject JSON files ---
    print("\nWriting per-subject JSON files...")
    for subj in sorted(VALID_SUBJECTS):
        if subj not in subject_questions:
            print(f"  [SKIP] {subj}: no questions found")
            continue
        fname = FILE_NAMES[subj] + ".json"
        outpath = os.path.join(OUTPUT_DIR, fname)
        # Merge all splits into a single list, adding a "split" field
        all_qs = []
        for split_name in ["dev", "test", "train"]:
            for q in subject_questions[subj].get(split_name, []):
                q_copy = dict(q)
                q_copy["split"] = split_name
                all_qs.append(q_copy)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(all_qs, f, ensure_ascii=False, indent=2)
        print(f"  {fname}: {len(all_qs)} questions")

    # --- Generate Markdown Report ---
    print("\nGenerating report...")

    total_all = sum(split_totals.values())
    total_filtered = sum(len(v) for v in filtered_out.values())
    total_valid = total_all - total_filtered

    lines = []
    lines.append("# MedMCQA Dataset - Subject Split Report\n")
    lines.append("## Overview\n")
    lines.append(f"- **Total questions in dataset**: {total_all:,}")
    lines.append(f"- **Valid questions (in known subjects)**: {total_valid:,}")
    lines.append(f"- **Filtered out questions**: {total_filtered:,}")
    lines.append(f"- **Number of subjects**: {len(VALID_SUBJECTS)}\n")

    lines.append("### Questions per Source Split\n")
    lines.append("| Split | Total | Valid | Filtered |")
    lines.append("|-------|------:|------:|---------:|")
    for split_name in ["dev", "test", "train"]:
        t = split_totals[split_name]
        filt = len(filtered_out.get(split_name, []))
        valid = t - filt
        lines.append(f"| {split_name} | {t:,} | {valid:,} | {filt:,} |")
    lines.append("")

    # Filtered out details
    if total_filtered > 0:
        lines.append("### Filtered Out Questions Detail\n")
        filtered_subjects = Counter()
        for split_name, items in filtered_out.items():
            for item in items:
                s = item.get("subject_name") or "null/empty"
                filtered_subjects[s] += 1
        lines.append("| Subject Name | Count |")
        lines.append("|-------------|------:|")
        for s, c in filtered_subjects.most_common():
            lines.append(f"| {s} | {c:,} |")
        lines.append("")

    # Main table
    lines.append("## Per-Subject Breakdown\n")
    lines.append("| # | Subject (Dataset) | Display Name | Dev | Test | Train | **Total** | Single | Multi | With Exp | Without Exp | File |")
    lines.append("|--:|-------------------|-------------|----:|-----:|------:|----------:|-------:|------:|---------:|------------:|------|")

    idx = 0
    grand_dev = grand_test = grand_train = grand_total = 0
    grand_single = grand_multi = grand_wexp = grand_woexp = 0

    for subj in sorted(VALID_SUBJECTS):
        idx += 1
        display = DISPLAY_NAMES[subj]
        fname = FILE_NAMES[subj] + ".json"
        dev_c = len(subject_questions[subj].get("dev", []))
        test_c = len(subject_questions[subj].get("test", []))
        train_c = len(subject_questions[subj].get("train", []))
        total_c = dev_c + test_c + train_c
        single_c = subject_choice_types[subj].get("single", 0)
        multi_c = subject_choice_types[subj].get("multi", 0)
        wexp = subject_has_exp[subj].get("with_exp", 0)
        woexp = subject_has_exp[subj].get("without_exp", 0)

        grand_dev += dev_c
        grand_test += test_c
        grand_train += train_c
        grand_total += total_c
        grand_single += single_c
        grand_multi += multi_c
        grand_wexp += wexp
        grand_woexp += woexp

        lines.append(
            f"| {idx} | {subj} | {display} | {dev_c:,} | {test_c:,} | {train_c:,} | **{total_c:,}** | {single_c:,} | {multi_c:,} | {wexp:,} | {woexp:,} | `{fname}` |"
        )

    lines.append(
        f"| | **TOTAL** | | **{grand_dev:,}** | **{grand_test:,}** | **{grand_train:,}** | **{grand_total:,}** | **{grand_single:,}** | **{grand_multi:,}** | **{grand_wexp:,}** | **{grand_woexp:,}** | |"
    )
    lines.append("")

    # Data fields description
    lines.append("## Data Fields\n")
    lines.append("Each question in the output JSON files contains the following fields:\n")
    lines.append("| Field | Description |")
    lines.append("|-------|-------------|")
    lines.append("| `id` | Unique string identifier for the question |")
    lines.append("| `question` | Question text |")
    lines.append("| `opa` | Option A |")
    lines.append("| `opb` | Option B |")
    lines.append("| `opc` | Option C |")
    lines.append("| `opd` | Option D |")
    lines.append("| `cop` | Correct option (1=A, 2=B, 3=C, 4=D) — present in dev and train splits |")
    lines.append("| `choice_type` | `single` or `multi` choice question |")
    lines.append("| `exp` | Expert's explanation (may be null) |")
    lines.append("| `subject_name` | Medical subject name |")
    lines.append("| `topic_name` | Medical topic name (may be null) |")
    lines.append("| `split` | Original split: `dev`, `test`, or `train` |")
    lines.append("")

    lines.append("## Output Structure\n")
    lines.append("```")
    lines.append("MedMCQA/")
    lines.append("├── data/")
    lines.append("│   ├── dev.json           (original)")
    lines.append("│   ├── test.json          (original)")
    lines.append("│   ├── train.json         (original)")
    lines.append("│   └── by_subject/")
    for subj in sorted(VALID_SUBJECTS):
        fname = FILE_NAMES[subj] + ".json"
        total_c = sum(len(subject_questions[subj].get(s, [])) for s in ["dev", "test", "train"])
        lines.append(f"│       ├── {fname}  ({total_c:,} questions)")
    lines.append("├── split_by_subject.py")
    lines.append("└── report.md")
    lines.append("```\n")

    report_text = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"\nDone! Report saved to: {REPORT_PATH}")
    print(f"Subject JSON files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
