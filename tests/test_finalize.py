import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finalize import finalize


def test_final_status_requires_complete_patch_control():
    config = {"control": {"max_hit_gap": 0.05}}
    stats = {"treatment": "proposed", "control": "baseline", "seeds": [1]}
    preliminary = {"bdd": {"hypothesis_confirmed": True, "usefulness_level": 0.5}}
    row = {"dataset": "bdd", "split": "test", "fill": "bg", "seed": 1, "patches": 100}
    baseline = {**row, "regime": "baseline", "patches_hit": 0}
    proposed = {**row, "regime": "proposed", "patches_hit": 6}
    [missing] = finalize(preliminary, [baseline], config, stats, "test")
    assert missing["status"] == "incomplete"
    assert missing["hypothesis_confirmed"] is None
    [blocked] = finalize(preliminary, [baseline, proposed], config, stats, "test")
    assert blocked["status"] == "blocked_by_patch_control"
    assert blocked["usefulness_level"] is None
