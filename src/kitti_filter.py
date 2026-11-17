"""Отбор объектов KITTI: Cyclist -> occlusion = 0 -> truncation <= порога -> высота рамки.

Формат строки label_2: type truncated occluded alpha left top right bottom h w l x y z rot_y.

Выход (reports/):
  kitti_counts.csv          изображения и объекты до/после каждого шага
  kitti_selected.csv        отобранные объекты (occlusion = 0)
  kitti_selected_occ1.csv   то же для occlusion = 1 (диагностика)
  kitti_occ_trunc.csv       распределение occlusion по Cyclist
  kitti_filter.md           сводка и сравнение с введением

Запуск: python src/kitti_filter.py [--config configs/kitti_filter.yaml] [--limit N]
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml

STEPS = ["all_images", "cyclist", "occlusion", "truncation", "min_height"]


def parse(line):
    p = line.split()
    return {
        "type": p[0], "truncated": float(p[1]), "occluded": int(p[2]),
        "left": float(p[4]), "top": float(p[5]), "right": float(p[6]), "bottom": float(p[7]),
    }


def steps_passed(o, occlusion, f):
    """Шаги, пройденные объектом Cyclist при заданном уровне occlusion."""
    passed = []
    if o["occluded"] != occlusion:
        return passed
    passed.append("occlusion")
    if o["truncated"] > f["max_truncation"]:
        return passed
    passed.append("truncation")
    if o["bottom"] - o["top"] >= f["min_height_px"]:
        passed.append("min_height")
    return passed


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kitti_filter.yaml")
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N файлов разметки")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    f = cfg["filters"]
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(cfg["label_dir"]).glob("*.txt"))
    if args.limit:
        files = files[:args.limit]

    obj = {k: Counter() for k in ("main", "extra")}
    imgs = {k: {s: set() for s in STEPS} for k in ("main", "extra")}
    occ_dist = Counter()
    selected = {"main": [], "extra": []}
    levels = {"main": f["occlusion"], "extra": cfg["extra_occlusion_subset"]}

    for path in files:
        image = path.stem
        for k in imgs:
            imgs[k]["all_images"].add(image)
        for idx, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            o = parse(line)
            if o["type"] != cfg["category"]:
                continue
            occ_dist[o["occluded"]] += 1
            for k, level in levels.items():
                passed = ["cyclist"] + steps_passed(o, level, f)
                for s in passed:
                    obj[k][s] += 1
                    imgs[k][s].add(image)
                if passed[-1] == "min_height":
                    selected[k].append({
                        "image": image, "line": idx, "truncated": o["truncated"], "occluded": o["occluded"],
                        "left": o["left"], "top": o["top"], "right": o["right"], "bottom": o["bottom"],
                        "height": round(o["bottom"] - o["top"], 2),
                    })

    fields = ["subset", "step", "images", "objects"]
    counts = [{"subset": f"occlusion={levels[k]}", "step": s, "images": len(imgs[k][s]),
               "objects": obj[k][s] if s != "all_images" else ""} for k in ("main", "extra") for s in STEPS]
    write_csv(out / "kitti_counts.csv", counts, fields)
    sel_fields = ["image", "line", "truncated", "occluded", "left", "top", "right", "bottom", "height"]
    write_csv(out / "kitti_selected.csv", selected["main"], sel_fields)
    write_csv(out / f"kitti_selected_occ{levels['extra']}.csv", selected["extra"], sel_fields)
    write_csv(out / "kitti_occ_trunc.csv",
              [{"occluded": k, "objects": v} for k, v in sorted(occ_dist.items())], ["occluded", "objects"])

    for r in counts:
        print(f"{r['subset']:12s} {r['step']:11s} images={r['images']:5d} objects={r['objects']}")

    lines = ["# KITTI: отбор Cyclist", "",
             f"occlusion = {f['occlusion']}; truncation <= {f['max_truncation']}; высота >= {f['min_height_px']} px. "
             f"Подмножество occlusion = {levels['extra']} — с теми же порогами truncation и высоты.", "",
             "| подмножество | шаг | изображений | объектов |", "|---|---|---|---|"]
    lines += [f"| {r['subset']} | {r['step']} | {r['images']} | {r['objects']} |" for r in counts]
    lines += ["", "## Сравнение с введением", "", "| шаг | введение | сейчас | разница |", "|---|---|---|---|"]
    for step, val in cfg.get("reference_counts", {}).items():
        now = obj["main"][step]
        lines.append(f"| {step} | {val} | {now} | {now - val:+d} |")
    lines += ["", "## Распределение occlusion у всех Cyclist", "", "| occlusion | объектов |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(occ_dist.items())]
    (out / "kitti_filter.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Отчёты: {out}/kitti_*.csv, {out}/kitti_filter.md")


if __name__ == "__main__":
    main()
