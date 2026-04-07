"""
Step 3: Compare classifications from two models and select the final 200 questions.

- Computes inter-model agreement (accordance rate)
- Selects 50 questions per category, prioritizing:
  1. Both models agree on category (high accordance)
  2. Both models have high confidence
  3. Diverse subject representation within each category
- Generates a detailed report

Usage:
    python compare_and_select.py
"""

import json
import random
from collections import Counter, defaultdict
from config import (
    CATEGORIES,
    MODEL_CONFIGS,
    OUTPUT_DIR,
    RANDOM_SEED,
)

FINAL_PER_CATEGORY = 50
FINAL_TOTAL = FINAL_PER_CATEGORY * len(CATEGORIES)  # 200


def load_classified(model_key):
    path = OUTPUT_DIR / f"classified_{model_key}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}\nRun classify_candidates.py --model {model_key} first.")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_agreement(data_a, data_b):
    """Compare classifications question-by-question using 'id' as key."""
    # Index by question ID
    map_a = {q["id"]: q for q in data_a}
    map_b = {q["id"]: q for q in data_b}

    common_ids = set(map_a.keys()) & set(map_b.keys())
    if not common_ids:
        raise ValueError("No common question IDs found between the two model outputs.")

    agree = 0
    disagree = 0
    results = []

    for qid in common_ids:
        qa = map_a[qid]
        qb = map_b[qid]
        cat_a = qa["classification"]["category"]
        cat_b = qb["classification"]["category"]
        conf_a = qa["classification"]["confidence"]
        conf_b = qb["classification"]["confidence"]

        agreed = cat_a == cat_b
        if agreed:
            agree += 1
        else:
            disagree += 1

        results.append({
            "id": qid,
            "question": qa.get("question", ""),
            "subject_name": qa.get("subject_name", ""),
            "cat_model_a": cat_a,
            "cat_model_b": cat_b,
            "conf_model_a": conf_a,
            "conf_model_b": conf_b,
            "agreed": agreed,
            # Keep full question data from model_a as base
            "_full": qa,
        })

    return results, agree, disagree


def select_final(comparison_results):
    """Select 50 questions per category with prioritization."""
    random.seed(RANDOM_SEED)

    # Group by agreed category
    agreed_by_cat = defaultdict(list)
    disagreed = []

    for r in comparison_results:
        if r["agreed"]:
            agreed_by_cat[r["cat_model_a"]].append(r)
        else:
            disagreed.append(r)

    final_selection = {}
    selection_stats = {}

    for cat in CATEGORIES:
        pool = agreed_by_cat.get(cat, [])

        # Sort by confidence: both high > one high > both medium > rest
        def confidence_score(r):
            score = 0
            for conf in [r["conf_model_a"], r["conf_model_b"]]:
                if conf == "high":
                    score += 2
                elif conf == "medium":
                    score += 1
            return score

        pool.sort(key=confidence_score, reverse=True)

        # Select with subject diversity
        selected = []
        subject_count = Counter()

        # First pass: pick highest confidence, spread across subjects
        for r in pool:
            subj = r["subject_name"]
            # Allow max ~ceil(50/num_subjects_in_pool) per subject initially
            unique_subjects = len(set(x["subject_name"] for x in pool))
            max_per_subj_initial = max(3, 50 // max(unique_subjects, 1) + 2)
            if subject_count[subj] < max_per_subj_initial and len(selected) < FINAL_PER_CATEGORY:
                selected.append(r)
                subject_count[subj] += 1

        # Second pass: fill remaining slots regardless of subject balance
        if len(selected) < FINAL_PER_CATEGORY:
            selected_ids = {r["id"] for r in selected}
            for r in pool:
                if r["id"] not in selected_ids and len(selected) < FINAL_PER_CATEGORY:
                    selected.append(r)

        # If still short, pull from disagreed where at least one model said this category
        if len(selected) < FINAL_PER_CATEGORY:
            selected_ids = {r["id"] for r in selected}
            for r in disagreed:
                if len(selected) >= FINAL_PER_CATEGORY:
                    break
                if r["id"] in selected_ids:
                    continue
                if r["cat_model_a"] == cat or r["cat_model_b"] == cat:
                    # Mark as needing manual review
                    r["needs_manual_review"] = True
                    selected.append(r)

        final_selection[cat] = selected
        selection_stats[cat] = {
            "selected": len(selected),
            "from_agreed": sum(1 for s in selected if s["agreed"]),
            "from_disagreed": sum(1 for s in selected if not s["agreed"]),
            "subjects": dict(Counter(s["subject_name"] for s in selected)),
            "high_conf_both": sum(1 for s in selected if s["conf_model_a"] == "high" and s["conf_model_b"] == "high"),
        }

    return final_selection, selection_stats


def generate_report(comparison_results, agree, disagree, final_selection, selection_stats, model_a_name, model_b_name):
    """Generate a comprehensive markdown report."""
    total = agree + disagree
    rate = (agree / total * 100) if total > 0 else 0

    lines = []
    lines.append("# Classification Comparison & Selection Report\n")

    # Models
    lines.append("## Models Used\n")
    lines.append(f"- **Model A**: {model_a_name}")
    lines.append(f"- **Model B**: {model_b_name}\n")

    # Overall agreement
    lines.append("## Inter-Model Agreement\n")
    lines.append(f"- **Total questions compared**: {total}")
    lines.append(f"- **Agreed**: {agree} ({rate:.1f}%)")
    lines.append(f"- **Disagreed**: {disagree} ({100-rate:.1f}%)\n")

    # Agreement by category
    lines.append("### Agreement by Category\n")
    lines.append("| Category | Agreed | Model A only | Model B only |")
    lines.append("|----------|-------:|-------------:|-------------:|")

    for cat in CATEGORIES:
        agreed_count = sum(1 for r in comparison_results if r["agreed"] and r["cat_model_a"] == cat)
        a_only = sum(1 for r in comparison_results if not r["agreed"] and r["cat_model_a"] == cat)
        b_only = sum(1 for r in comparison_results if not r["agreed"] and r["cat_model_b"] == cat)
        lines.append(f"| {cat} | {agreed_count} | {a_only} | {b_only} |")
    lines.append("")

    # Confusion matrix
    lines.append("### Confusion Matrix (Model A rows × Model B columns)\n")
    header = "| | " + " | ".join(CATEGORIES) + " |"
    sep = "|---|" + "|".join(["---:"] * len(CATEGORIES)) + "|"
    lines.append(header)
    lines.append(sep)
    for cat_a in CATEGORIES:
        row = f"| **{cat_a}** |"
        for cat_b in CATEGORIES:
            count = sum(1 for r in comparison_results if r["cat_model_a"] == cat_a and r["cat_model_b"] == cat_b)
            row += f" {count} |"
        lines.append(row)
    lines.append("")

    # Final selection
    lines.append("## Final Selection (200 Questions)\n")
    lines.append("| Category | Selected | From Agreed | From Disagreed | High Conf (both) |")
    lines.append("|----------|--------:|------------:|---------------:|------------------:|")
    total_sel = 0
    for cat in CATEGORIES:
        s = selection_stats[cat]
        total_sel += s["selected"]
        lines.append(
            f"| {cat} | {s['selected']} | {s['from_agreed']} | {s['from_disagreed']} | {s['high_conf_both']} |"
        )
    lines.append(f"| **TOTAL** | **{total_sel}** | | | |")
    lines.append("")

    # Subject distribution within each category
    lines.append("### Subject Distribution per Category\n")
    for cat in CATEGORIES:
        lines.append(f"#### {cat}\n")
        subjects = selection_stats[cat]["subjects"]
        lines.append("| Subject | Count |")
        lines.append("|---------|------:|")
        for subj, count in sorted(subjects.items(), key=lambda x: -x[1]):
            lines.append(f"| {subj} | {count} |")
        lines.append("")

    # Disagreement samples (for manual review)
    disagreed_items = [r for r in comparison_results if not r["agreed"]]
    if disagreed_items:
        lines.append("## Disagreement Samples (first 20)\n")
        lines.append("| # | Subject | Model A | Model B | Question (truncated) |")
        lines.append("|--:|---------|---------|---------|---------------------|")
        for i, r in enumerate(disagreed_items[:20], 1):
            q_short = r["question"][:80] + "..." if len(r["question"]) > 80 else r["question"]
            lines.append(f"| {i} | {r['subject_name']} | {r['cat_model_a']} | {r['cat_model_b']} | {q_short} |")
        lines.append("")

    return "\n".join(lines)


def main():
    random.seed(RANDOM_SEED)

    model_keys = list(MODEL_CONFIGS.keys())
    if len(model_keys) < 2:
        print("ERROR: Need at least 2 model configurations in config.py")
        return

    model_a_key, model_b_key = model_keys[0], model_keys[1]
    model_a_name = MODEL_CONFIGS[model_a_key]["model_name"]
    model_b_name = MODEL_CONFIGS[model_b_key]["model_name"]

    print(f"Loading classifications...")
    print(f"  Model A: {model_a_name} ({model_a_key})")
    print(f"  Model B: {model_b_name} ({model_b_key})")

    data_a = load_classified(model_a_key)
    data_b = load_classified(model_b_key)

    print(f"  Model A: {len(data_a)} questions")
    print(f"  Model B: {len(data_b)} questions")

    # Compare
    print("\nComputing agreement...")
    comparison, agree, disagree = compute_agreement(data_a, data_b)
    total = agree + disagree
    rate = (agree / total * 100) if total > 0 else 0
    print(f"  Agreement: {agree}/{total} ({rate:.1f}%)")

    # Select final 200
    print("\nSelecting final 200 questions...")
    final_selection, selection_stats = select_final(comparison)

    for cat in CATEGORIES:
        s = selection_stats[cat]
        print(f"  {cat}: {s['selected']} selected ({s['from_agreed']} agreed, {s['from_disagreed']} disagreed)")

    # Save final selection as JSONL (one file per category + one combined)
    final_all = []
    for cat in CATEGORIES:
        cat_questions = []
        for r in final_selection[cat]:
            q = dict(r["_full"])
            q["macro_category"] = cat
            q["model_agreement"] = r["agreed"]
            q["needs_manual_review"] = r.get("needs_manual_review", False)
            cat_questions.append(q)
            final_all.append(q)

        # Per-category file
        cat_fname = cat.lower().replace(" ", "_").replace("&", "and")
        cat_path = OUTPUT_DIR / f"final_{cat_fname}.jsonl"
        with open(cat_path, "w", encoding="utf-8") as f:
            for q in cat_questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        print(f"  Saved {len(cat_questions)} → {cat_path.name}")

    # Combined file
    combined_path = OUTPUT_DIR / "final_200_questions.jsonl"
    with open(combined_path, "w", encoding="utf-8") as f:
        for q in final_all:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\n  Combined → {combined_path.name} ({len(final_all)} questions)")

    # Generate report
    report = generate_report(comparison, agree, disagree, final_selection, selection_stats, model_a_name, model_b_name)
    report_path = OUTPUT_DIR / "comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report  → {report_path.name}")

    # Also save full comparison data for auditing
    audit_path = OUTPUT_DIR / "full_comparison.jsonl"
    with open(audit_path, "w", encoding="utf-8") as f:
        for r in comparison:
            row = {k: v for k, v in r.items() if k != "_full"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Audit   → {audit_path.name}")


if __name__ == "__main__":
    main()
