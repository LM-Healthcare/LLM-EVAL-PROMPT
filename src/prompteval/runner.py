"""Resumable experiment runner.

One task = (model, item language, condition, item, run). Every completed task
is appended as one JSON line to <output_dir>/responses/<model_id>.jsonl, so an
interrupted run resumes where it stopped and re-running never duplicates calls.
Transient API failures are retried with exponential backoff; a request error
(e.g. HTTP 400 from an unsupported parameter) stops that model immediately
instead of burning through thousands of failing calls.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from . import __version__
from .backends import FatalError, RetryableError, make_backend
from .backends.mock_backend import MockBackend
from .config import output_dir
from .dataset import Item, dataset_commit, select_items
from .prompts import PromptBank, load_best, resolve_condition


@dataclass(frozen=True)
class Task:
    model_id: str
    lang: str
    condition_alias: str
    condition: str
    code: str
    run: int

    @property
    def key(self) -> str:
        return f"{self.model_id}|{self.lang}|{self.condition}|{self.code}|{self.run}"


def prompt_language(cfg: dict, item_lang: str) -> str:
    pl = cfg["prompts"]["language"]
    return item_lang if pl == "match" else pl


def build_tasks(cfg: dict, items: list[Item], model_ids: list[str] | None = None) -> list[Task]:
    best = load_best(cfg["prompts"].get("best_file"))
    best_map = cfg["prompts"].get("best_model_map") or {}  # e.g. {sonnet55_thinking: sonnet55}
    runs = int(cfg["sampling"]["runs"])
    tasks = []
    for m in cfg["models"]:
        if model_ids and m["id"] not in model_ids:
            continue
        for lang in cfg["dataset"]["languages"]:
            for alias in cfg["prompts"]["conditions"]:
                cond = resolve_condition(alias, best_map.get(m["id"], m["id"]), best)
                for run in range(1, runs + 1):
                    for it in items:
                        tasks.append(Task(m["id"], lang, alias, cond, it.code, run))
    if cfg["runner"].get("order") == "item_major":
        tasks.sort(key=lambda t: (t.model_id, t.code, t.lang, t.condition, t.run))
    return tasks  # default: run-major, so partial results stay balanced across runs


def completed_keys(path: Path) -> set[str]:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial line from a crash: the task is simply redone
                if rec.get("status") == "ok":
                    done.add(rec["key"])
    return done


class RateLimiter:
    def __init__(self, rpm: float | None):
        self.interval = 60.0 / rpm if rpm else 0.0
        self.lock = threading.Lock()
        self.next_time = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_time)
            self.next_time = t + self.interval
        if t > now:
            time.sleep(t - now)


def _seed_for(cfg: dict, mcfg: dict, run: int) -> int | None:
    if not mcfg.get("seed_per_run"):
        return None
    return int(cfg["experiment"]["seed"]) * 1000 + run


def write_manifest(cfg: dict, out: Path, items: list[Item], bank: PromptBank) -> None:
    manifest = {
        "written": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "prompteval_version": __version__,
        "python": platform.python_version(),
        "config_path": cfg.get("_config_path"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "dataset_commit": dataset_commit(cfg["dataset"]["path"]),
        "prompt_file_hash": bank.file_hash,
        "prompt_version": bank.version,
        "n_items": len(items),
        "item_codes": [i.code for i in items],
    }
    out.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / "manifests").mkdir(exist_ok=True)
    with open(out / "manifests" / f"manifest_{stamp}.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def run_experiment(cfg: dict, model_ids: list[str] | None = None, dry_run: bool = False,
                   max_tasks: int | None = None) -> dict:
    items = select_items(cfg)
    by_code = {i.code: i for i in items}
    bank = PromptBank(cfg["prompts"]["file"])
    out = output_dir(cfg)
    tasks = build_tasks(cfg, items, model_ids)
    shuffle = cfg["dataset"].get("shuffle_options", True)
    sseed = int(cfg["dataset"].get("shuffle_seed", 2026))
    ds_root = Path(cfg["dataset"]["path"]) / "Dataset"

    summary = {"items": len(items), "tasks_total": len(tasks), "per_model": {}}

    if dry_run:
        ddir = out / "dry_run"
        ddir.mkdir(parents=True, exist_ok=True)
        seen = set()
        for t in tasks:
            sig = (t.model_id, t.lang, t.condition)
            if sig in seen:
                continue
            seen.add(sig)
            it = by_code[t.code]
            perm = it.permutation(sseed, shuffle)
            rp = bank.render(it, t.lang, prompt_language(cfg, t.lang), t.condition, perm)
            name = f"{t.model_id}_{t.lang}_{t.condition.replace('+', '-')}.txt"
            (ddir / name).write_text(f"[SYSTEM]\n{rp.system}\n\n[USER]\n{rp.user}\n", encoding="utf-8")
        for m in cfg["models"]:
            if model_ids and m["id"] not in model_ids:
                continue
            mt = [t for t in tasks if t.model_id == m["id"]]
            done = completed_keys(out / "responses" / f"{m['id']}.jsonl")
            summary["per_model"][m["id"]] = {"tasks": len(mt), "done": sum(t.key in done for t in mt)}
        summary["example_prompts"] = str(ddir)
        return summary

    write_manifest(cfg, out, items, bank)
    (out / "responses").mkdir(parents=True, exist_ok=True)
    prompts_path = out / "prompts.jsonl"
    known_prompts = set()
    if prompts_path.exists():
        with open(prompts_path, encoding="utf-8") as f:
            known_prompts = {json.loads(l)["prompt_hash"] for l in f if l.strip()}
    plock = threading.Lock()
    rc = cfg["runner"]

    for mcfg in cfg["models"]:
        if model_ids and mcfg["id"] not in model_ids:
            continue
        resp_path = out / "responses" / f"{mcfg['id']}.jsonl"
        done = completed_keys(resp_path)
        todo = [t for t in tasks if t.model_id == mcfg["id"] and t.key not in done]
        if max_tasks:
            todo = todo[:max_tasks]
        summary["per_model"][mcfg["id"]] = {"already_done": len(done), "to_run": len(todo)}
        if not todo:
            continue
        backend = make_backend(mcfg)
        static = backend.describe()
        limiter = RateLimiter(mcfg.get("requests_per_minute"))
        stop = threading.Event()
        wlock = threading.Lock()
        fatal: list[str] = []
        stats = {"ok": 0, "failed": 0}

        def work(t: Task, mcfg=mcfg, backend=backend, static=static, limiter=limiter,
                 stop=stop, resp_path=resp_path, wlock=wlock, fatal=fatal, stats=stats):
            if stop.is_set():
                return
            it = by_code[t.code]
            perm = it.permutation(sseed, shuffle)
            _, correct_letter = it.presented(t.lang, perm)
            plang = prompt_language(cfg, t.lang)
            rp = bank.render(it, t.lang, plang, t.condition, perm)
            with plock:
                if rp.prompt_hash not in known_prompts:
                    known_prompts.add(rp.prompt_hash)
                    with open(prompts_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"prompt_hash": rp.prompt_hash, "system": rp.system,
                                            "user": rp.user}, ensure_ascii=False) + "\n")
            image = str(ds_root / it.image_path) if (it.has_image and it.image_path) else None
            seed = _seed_for(cfg, mcfg, t.run)
            attempt, delay, last_err = 0, float(rc["retry_base_delay"]), ""
            while not stop.is_set():
                limiter.wait()
                t0 = time.monotonic()
                try:
                    if isinstance(backend, MockBackend):
                        gen = backend.generate(rp.system, rp.user, image, seed, code=t.code,
                                               condition=t.condition, lang=t.lang,
                                               correct=correct_letter)
                    else:
                        gen = backend.generate(rp.system, rp.user, image, seed)
                    break
                except RetryableError as e:
                    attempt += 1
                    last_err = str(e)[:500]
                    if attempt > int(rc["max_retries"]):
                        with wlock:
                            stats["failed"] += 1
                            with open(out / "errors.jsonl", "a", encoding="utf-8") as f:
                                f.write(json.dumps({"key": t.key, "error": last_err,
                                                    "time": dt.datetime.now().isoformat()}) + "\n")
                        return
                    time.sleep(min(delay, float(rc["retry_max_delay"])) * (0.5 + random.random()))
                    delay *= 2
                except FatalError as e:
                    fatal.append(str(e))
                    stop.set()
                    return
            else:
                return
            rec = {
                "key": t.key,
                "status": "ok",
                "experiment": cfg["experiment"]["name"],
                "model_id": t.model_id,
                "language": t.lang,
                "prompt_language": plang,
                "condition": t.condition,
                "condition_alias": t.condition_alias,
                "question_code": t.code,
                "run": t.run,
                "seed": seed,
                "option_order": perm,
                "correct_letter": correct_letter,
                "prompt_hash": rp.prompt_hash,
                "prompt_file_hash": bank.file_hash,
                "response_text": gen.text,
                "thinking_text": gen.thinking,
                "finish_reason": gen.finish_reason,
                "model_returned": gen.model_returned,
                "usage": gen.usage,
                "request": static,
                "latency_s": round(time.monotonic() - t0, 3),
                "attempts": attempt + 1,
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            }
            with wlock:
                with open(resp_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["ok"] += 1

        workers = int(mcfg.get("concurrency", 4))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(work, t) for t in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=mcfg["id"], unit="call"):
                try:
                    fut.result()
                except Exception as e:  # unexpected bug: record it and stop this model
                    fatal.append(f"{type(e).__name__}: {e}")
                    stop.set()
        summary["per_model"][mcfg["id"]].update(stats)
        if fatal:
            summary["per_model"][mcfg["id"]]["fatal_error"] = fatal[0]
    return summary
