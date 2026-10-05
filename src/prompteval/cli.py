"""Command-line interface: `prompteval <command> ...`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _parsed_from_dirs(dirs: list[str]) -> pd.DataFrame:
    from .analysis import parse_frame, read_responses

    frames = [read_responses(Path(d)) for d in dirs]
    frames = [f for f in frames if len(f)]
    if not frames:
        sys.exit(f"No responses found in {dirs}")
    return parse_frame(pd.concat(frames, ignore_index=True))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="prompteval", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch-dataset", help="clone ITAMed (optionally at a pinned commit)")
    p.add_argument("--dest", default="data/ITAMed")
    p.add_argument("--ref", default=None, help="commit / tag to check out")

    p = sub.add_parser("make-split", help="create the frozen evaluation / fine-tuning split")
    p.add_argument("--dataset", default="data/ITAMed")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-per-stratum", type=int, default=1)
    p.add_argument("--year-weight", action="append", default=[], metavar="YEAR=W",
                   help="over-weight a year inside each specialty, e.g. 2025=1.5 (repeatable)")
    p.add_argument("--question-types", nargs="+", default=["case-based"])
    p.add_argument("--include-images", action="store_true")
    p.add_argument("--ft-val-fraction", type=float, default=0.1)
    p.add_argument("--out", default="data/splits/eval_split.json")
    p.add_argument("--force", action="store_true")

    for name, hlp in (("dry-run", "count tasks and render example prompts, no API calls"),
                      ("run", "run (or resume) an experiment"),
                      ("analyze", "compute metrics, contrasts and report"),
                      ("select-best", "pick the best condition per model (a priori rule)"),
                      ("status", "progress of an experiment")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("-c", "--config", required=True)
        if name in ("run", "dry-run"):
            p.add_argument("--model", nargs="+", default=None, help="only these model ids")
        if name == "run":
            p.add_argument("--max-tasks", type=int, default=None, help="stop after N new calls per model")
        if name == "select-best":
            p.add_argument("--out", default=None)

    p = sub.add_parser("export-distractors", help="rating sheets for chosen distractors")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--from", dest="dirs", nargs="+", default=None,
                   help="experiment output dirs to pool (default: this experiment)")
    p.add_argument("--raters", nargs="+", default=["R1", "R2"])
    p.add_argument("--lang", default="it", choices=["it", "en"])
    p.add_argument("--out", default="annotations/distractors")
    p.add_argument("--only-new", action="store_true",
                   help="skip distractors already in the final ratings file")

    p = sub.add_parser("import-distractors", help="consolidate distractor ratings")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--adjudication", default=None)
    p.add_argument("--out", default="annotations/distractor_harm_final.csv")

    p = sub.add_parser("export-urgency", help="gold-urgency sheets for the evaluation items")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--raters", nargs="+", default=["R1", "R2"])
    p.add_argument("--lang", default="it", choices=["it", "en"])
    p.add_argument("--out", default="annotations/urgency")

    p = sub.add_parser("import-urgency", help="consolidate gold-urgency labels")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--adjudication", default=None)
    p.add_argument("--out", default="annotations/urgency_final.csv")

    p = sub.add_parser("export-references", help="sample of C3 references to verify")
    p.add_argument("-c", "--config", required=True)
    p.add_argument("--per-model", type=int, default=150)
    p.add_argument("--out", default="annotations/references/references_to_verify.csv")

    a = ap.parse_args(argv)

    if a.cmd == "fetch-dataset":
        from .dataset import fetch_dataset
        _print({"dataset": a.dest, "commit": fetch_dataset(a.dest, a.ref)})
        return

    if a.cmd == "make-split":
        from .dataset import load_itamed
        from .split import make_split, write_split
        items = load_itamed(a.dataset)
        yw = {int(k): float(v) for k, v in (x.split("=") for x in a.year_weight)}
        split = make_split(items, a.n, a.seed,
                           {"question_types": a.question_types, "include_images": a.include_images},
                           a.min_per_stratum, yw, a.ft_val_fraction, a.dataset)
        write_split(split, a.out, a.force)
        _print({"written": a.out, **split["summary"]})
        return

    if a.cmd in ("import-distractors", "import-urgency"):
        from .annotations import import_distractor_ratings, import_urgency
        fn = import_distractor_ratings if a.cmd == "import-distractors" else import_urgency
        _print(fn([Path(f) for f in a.files], Path(a.out),
                  Path(a.adjudication) if a.adjudication else None))
        return

    from .config import load_config, output_dir
    cfg = load_config(a.config)

    if a.cmd in ("run", "dry-run"):
        from .runner import run_experiment
        _print(run_experiment(cfg, a.model, dry_run=(a.cmd == "dry-run"),
                              max_tasks=getattr(a, "max_tasks", None)))
    elif a.cmd == "status":
        from .runner import run_experiment
        _print(run_experiment(cfg, None, dry_run=True))
    elif a.cmd == "analyze":
        from .analysis import analyse
        _print(analyse(cfg))
    elif a.cmd == "select-best":
        from .analysis import select_best
        out = a.out or str(output_dir(cfg) / "best_conditions.json")
        _print(select_best(cfg, out))
    elif a.cmd == "export-distractors":
        from .annotations import export_distractor_sheets
        from .dataset import load_itamed
        parsed = _parsed_from_dirs(a.dirs or [str(output_dir(cfg))])
        items = load_itamed(cfg["dataset"]["path"])
        existing = cfg["analysis"]["annotations"].get("distractor_ratings") if a.only_new else None
        _print(export_distractor_sheets(parsed, items, Path(a.out), a.raters, a.lang,
                                        existing=Path(existing) if existing else None))
    elif a.cmd == "export-urgency":
        from .annotations import export_urgency_sheets
        from .dataset import select_items
        _print(export_urgency_sheets(select_items(cfg), Path(a.out), a.raters, a.lang))
    elif a.cmd == "export-references":
        from .annotations import export_references
        parsed = _parsed_from_dirs([str(output_dir(cfg))])
        _print(export_references(parsed, Path(a.out), a.per_model))


if __name__ == "__main__":
    main()
