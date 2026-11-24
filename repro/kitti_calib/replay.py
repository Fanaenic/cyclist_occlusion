"""Повторный расчёт основного контраста технического пилота KITTI."""
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from finalize import finalize, write_final
from stats import analyse


def main():
    pcfg = yaml.safe_load((ROOT / "protocol.yaml").read_text(encoding="utf-8"))
    run = json.loads((ROOT / "run.json").read_text(encoding="utf-8"))
    stats = pcfg["stats"]
    argv = run["argv"]
    replicas = int(argv[argv.index("--replicas") + 1])
    override = {"datasets": ["kitti"], "split": "calib",
                "bootstrap": {**stats["bootstrap"], "replicas": replicas}}
    output = ROOT.parents[1] / "runs" / "replay_kitti_calib"
    _, decision = analyse(ROOT / "eval", pcfg, override, tag="kitti_calib",
                          report_dir=output, splits_dir=ROOT / "splits")
    with (ROOT / "control.csv").open(encoding="utf-8", newline="") as fh:
        controls = list(csv.DictReader(fh))
    for row in controls:
        for key in ("seed", "patches", "patches_hit"):
            row[key] = int(row[key])
    result = finalize(decision, controls, pcfg, stats, "calib")
    write_final(result, output)
    expected = json.loads((ROOT / "expected.json").read_text(encoding="utf-8"))
    if result != expected:
        raise SystemExit(f"Итог отличается от сохранённого: {output / 'final.json'}")
    print(f"Итог совпал с сохранённым: {output / 'final.json'}")


if __name__ == "__main__":
    main()
