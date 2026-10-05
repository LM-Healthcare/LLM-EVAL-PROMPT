import pytest

from prompteval.config import expand_env, load_config


def test_env_expansion(monkeypatch):
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    cfg = {"a": "${QWEN_BASE_URL:-http://localhost:8000/v1}", "b": ["x${NOPE:-y}"], "c": 3}
    assert expand_env(cfg) == {"a": "http://localhost:8000/v1", "b": ["xy"], "c": 3}
    monkeypatch.setenv("QWEN_BASE_URL", "http://vllm:8000/v1")
    assert expand_env(cfg)["a"] == "http://vllm:8000/v1"
    with pytest.raises(ValueError):
        expand_env("${SURELY_NOT_SET_VAR}")


def test_extends_does_not_inherit_output_dir(tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    cfg = load_config("configs/pilot.yaml")
    assert cfg["experiment"]["output_dir"] == "results/pilot"
    assert cfg["dataset"]["limit"] == 20 and cfg["sampling"]["runs"] == 2
    assert [m["id"] for m in cfg["models"]] == ["sonnet55", "qwen35_9b"]
    assert cfg["models"][1]["base_url"].startswith("http")


def test_all_shipped_configs_load(repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    for name in ("main", "pilot", "triage", "thinking_arm", "smoke_mock"):
        cfg = load_config(f"configs/{name}.yaml")
        assert cfg["experiment"]["output_dir"] == f"results/{name}"
