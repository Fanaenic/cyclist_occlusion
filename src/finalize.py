"""Согласованное решение по статистике и контролю заплаток."""
import csv
import json
from pathlib import Path


def finalize(statistical, controls, pcfg, scfg, part):
    results = []
    for dataset, decision in statistical.items():
        result = {"dataset": dataset, **decision}
        expected = {(regime, seed) for regime in (scfg["treatment"], scfg["control"])
                    for seed in scfg["seeds"]}
        rows = [r for r in controls if r["dataset"] == dataset and r["split"] == part
                and r["fill"] == "bg" and (r["regime"], r["seed"]) in expected]
        counts = {r["patches"] for r in rows}
        complete = (len(rows) == len(expected)
                    and {(r["regime"], r["seed"]) for r in rows} == expected
                    and len(counts) == 1
                    and all(0 <= r["patches_hit"] <= r["patches"] and r["patches"] > 0 for r in rows))
        gap = None
        if complete:
            rates = {regime: sum(r["patches_hit"] / r["patches"] for r in rows if r["regime"] == regime)
                     / len(scfg["seeds"]) for regime in (scfg["treatment"], scfg["control"])}
            gap = rates[scfg["treatment"]] - rates[scfg["control"]]
        result["control_hit_gap"] = gap
        result["shortcut_flag"] = gap is not None and gap > pcfg["control"]["max_hit_gap"]
        if not complete:
            status = "incomplete"
        elif result["shortcut_flag"]:
            status = "blocked_by_patch_control"
        elif decision["hypothesis_confirmed"] is None:
            status = "descriptive"
        elif decision["hypothesis_confirmed"]:
            status = "confirmed"
        else:
            status = "not_confirmed"
        result["status"] = status
        result["hypothesis_confirmed"] = {"confirmed": True, "not_confirmed": False}.get(status)
        if status in ("incomplete", "blocked_by_patch_control"):
            result["usefulness_level"] = None
        results.append(result)
    return results


def write_final(results, out):
    out = Path(out)
    (out / "final.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                    encoding="utf-8")
    with (out / "final.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Итог оценки", "", "| Датасет | Статус | Разность долей срабатываний на заплатках | Порог полезности |",
             "|---|---|---|---|"]
    for row in results:
        gap = row["control_hit_gap"]
        value = "нет данных" if gap is None else f"{gap:+.3f}"
        level = row["usefulness_level"]
        threshold = "не установлен" if level is None else f"{level:.0%}"
        lines.append(f"| {row['dataset']} | {row['status']} | {value} | {threshold} |")
    (out / "final.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
