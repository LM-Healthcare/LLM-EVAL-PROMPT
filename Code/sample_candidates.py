"""
Step 1: Sample candidate questions from the MedMCQA by-subject files.

- Only includes questions from dev and train splits (test has no correct answer).
- Samples proportionally from each subject with a minimum floor.
- Outputs a single JSON file with all candidates.
"""

import json
import random
from collections import Counter
from config import (
    BY_SUBJECT_DIR,
    CANDIDATES_FILE,
    RANDOM_SEED,
    TOTAL_CANDIDATES,
    MIN_PER_SUBJECT,
)


def load_subject_file(filepath):
    """Load a per-subject JSON file and filter to dev+train only."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Keep only dev and train (they have 'cop' = correct option)
    return [q for q in data if q.get("split") in ("dev", "train") and q.get("cop") is not None]


def compute_sample_sizes(subject_counts, total, minimum):
    """Compute how many questions to sample from each subject (proportional + floor)."""
    subjects = list(subject_counts.keys())
    grand_total = sum(subject_counts.values())

    # Start with proportional allocation
    raw = {s: max(minimum, int(total * subject_counts[s] / grand_total)) for s in subjects}

    # Cap at available count
    for s in subjects:
        raw[s] = min(raw[s], subject_counts[s])

    # Adjust to hit target total
    current_total = sum(raw.values())
    if current_total < total:
        # Distribute remainder proportionally among subjects with room
        remainder = total - current_total
        expandable = {s: subject_counts[s] - raw[s] for s in subjects if subject_counts[s] > raw[s]}
        exp_total = sum(expandable.values())
        if exp_total > 0:
            for s in expandable:
                extra = int(remainder * expandable[s] / exp_total)
                raw[s] += extra
    elif current_total > total:
        # Trim proportionally from subjects above minimum
        excess = current_total - total
        trimmable = {s: raw[s] - minimum for s in subjects if raw[s] > minimum}
        trim_total = sum(trimmable.values())
        if trim_total > 0:
            for s in trimmable:
                cut = min(int(excess * trimmable[s] / trim_total), trimmable[s])
                raw[s] -= cut

    return raw


def main():
    random.seed(RANDOM_SEED)

    # Discover all subject files
    subject_files = sorted(BY_SUBJECT_DIR.glob("*.json"))
    if not subject_files:
        print(f"ERROR: No subject files found in {BY_SUBJECT_DIR}")
        return

    # Load all subjects
    all_subjects = {}
    for fpath in subject_files:
        subject_name = fpath.stem.replace("_", " ")  # e.g. "Forensic Medicine"
        questions = load_subject_file(fpath)
        if questions:
            # Use the actual subject_name from the data
            actual_name = questions[0].get("subject_name", subject_name)
            all_subjects[actual_name] = questions
            print(f"  {actual_name}: {len(questions)} questions (dev+train)")

    print(f"\nTotal subjects: {len(all_subjects)}")
    total_available = sum(len(qs) for qs in all_subjects.values())
    print(f"Total available questions (dev+train): {total_available:,}")

    # Compute sample sizes
    subject_counts = {s: len(qs) for s, qs in all_subjects.items()}
    sample_sizes = compute_sample_sizes(subject_counts, TOTAL_CANDIDATES, MIN_PER_SUBJECT)

    print(f"\nSampling plan (target: {TOTAL_CANDIDATES}):")
    actual_total = 0
    for s in sorted(sample_sizes.keys()):
        n = sample_sizes[s]
        actual_total += n
        print(f"  {s}: {n} / {subject_counts[s]}")
    print(f"  TOTAL: {actual_total}")

    # Sample
    sampled = []
    for subject, questions in all_subjects.items():
        n = sample_sizes[subject]
        chosen = random.sample(questions, n)
        sampled.extend(chosen)

    # Shuffle the final set
    random.shuffle(sampled)

    # Save
    CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_FILE, "w", encoding="utf-8") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)

    print(f"\nSampled {len(sampled)} candidates → {CANDIDATES_FILE}")

    # Quick stats
    split_counts = Counter(q["split"] for q in sampled)
    choice_counts = Counter(q.get("choice_type", "unknown") for q in sampled)
    print(f"  By split: {dict(split_counts)}")
    print(f"  By choice_type: {dict(choice_counts)}")


if __name__ == "__main__":
    main()
