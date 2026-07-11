import json

from curvature_calib import provenance as prov

_REQUIRED_KEYS = {
    "experiment_id", "timestamp_utc", "git_commit", "git_short_commit",
    "git_dirty", "executed_command", "argv", "jax_version", "jaxlib_version",
    "numpy_version", "python_version", "platform", "hostname", "x64_enabled",
    "default_float_dtype", "seeds", "config", "plan",
}


def test_run_metadata_has_all_keys():
    md = prov.run_metadata("TEST", {"a": 1}, {"master": 0})
    assert _REQUIRED_KEYS.issubset(md.keys())


def test_run_metadata_records_config_and_seeds():
    md = prov.run_metadata("TEST", {"a": 1}, {"master": 7})
    assert md["config"] == {"a": 1}
    assert md["seeds"] == {"master": 7}


def test_run_metadata_records_command():
    md = prov.run_metadata("TEST", {}, {}, command="uv run python -m experiments.foo")
    assert md["executed_command"] == "uv run python -m experiments.foo"
    assert isinstance(md["argv"], list)


def test_write_json_roundtrips(tmp_path):
    md = prov.run_metadata("TEST", {"a": 1}, {"master": 0})
    p = tmp_path / "sub" / "provenance.json"
    prov.write_json(p, md)
    loaded = json.loads(p.read_text())
    assert loaded["experiment_id"] == "TEST"


def test_short_commit_is_string():
    assert isinstance(prov.short_commit(), str)
