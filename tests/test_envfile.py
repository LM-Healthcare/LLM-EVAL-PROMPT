import os

from prompteval.envfile import load_env_file, parse_env_file


def test_parse_env_file(tmp_path):
    f = tmp_path / ".env"
    f.write_text(
        "# comment\n"
        "ANTHROPIC_API_KEY=sk-ant-test\n"
        "export HF_TOKEN='hf_x'\n"
        'QUOTED="a # not a comment"\n'
        "INLINE=value # comment\n"
        "EMPTY=\n"
        "not a line\n",
        encoding="utf-8",
    )
    v = parse_env_file(f)
    assert v == {"ANTHROPIC_API_KEY": "sk-ant-test", "HF_TOKEN": "hf_x",
                 "QUOTED": "a # not a comment", "INLINE": "value", "EMPTY": ""}


def test_load_env_file_does_not_override(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("PE_TEST_A=from_file\nPE_TEST_B=from_file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PE_TEST_A", "from_shell")
    monkeypatch.setenv("PE_TEST_B", "")  # empty counts as unset
    path, loaded = load_env_file()
    assert path == tmp_path / ".env"
    assert os.environ["PE_TEST_A"] == "from_shell"
    assert os.environ["PE_TEST_B"] == "from_file"
    assert loaded == ["PE_TEST_B"]
