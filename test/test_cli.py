import sys
from ragql import cli
from ragql.config import Settings


def test_inline_query_from_config(monkeypatch, tmp_path, capsys):
    p1 = tmp_path / "a"
    p2 = tmp_path / "b"
    p1.mkdir()
    p2.mkdir()
    cfg = Settings(db_path=tmp_path / "db.sqlite")
    cfg.allowed_folders = [str(p1), str(p2)]

    record = {}

    class DummyRQ:
        def __init__(self, root, cfg_in):
            record["init"] = root

        def build(self):
            pass

        def query(self, q):
            record["question"] = q
            return "ANSWER"

    monkeypatch.setattr(cli.Settings, "load", classmethod(lambda cls: cfg))
    monkeypatch.setattr(cli, "RagQL", DummyRQ)
    monkeypatch.setattr(sys, "argv", ["ragql"])

    cli.main()

    out, _ = capsys.readouterr()
    assert "ANSWER" in out
    assert record["question"] == str(p2)
