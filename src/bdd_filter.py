"""Отбор объектов BDD100K: rider -> сопоставление с bike -> daytime -> высота рамки.

Сопоставление rider <-> bike — глобальный жадный алгоритм (см. configs/bdd_filter.yaml).
Для проверки замечания 3 параллельно считаются два других правила:
  - nonexclusive: rider сохраняется, если есть хоть один bike с IoU >= порога (так получены числа введения);
  - best_only: rider берёт только свой лучший bike и отбрасывается, если тот уже занят.

Выход (reports/):
  bdd_counts.csv            изображения и объекты до/после каждого шага, по частям
  bdd_algorithms.csv        сколько rider сопоставляет каждое из трёх правил
  bdd_ambiguous.csv         спорные случаи сопоставления
  bdd_occ_trunc.csv         распределение occluded/truncated после каждого шага
  bdd_selected_{split}.csv  отобранные пары rider-bike (вход для следующих шагов)
  bdd_filter.md             сводка и сравнение с введением

Запуск: python src/bdd_filter.py [--config configs/bdd_filter.yaml] [--limit N]
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

STEPS = ["all_images", "rider", "matched", "daytime", "min_height"]


def box_of(label):
    b = label["box2d"]
    return (b["x1"], b["y1"], b["x2"], b["y2"])


def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def target_box(rider, bike, mode):
    """Целевая рамка: rider — рамка человека; union — объединение рамок rider и его bike."""
    if mode == "rider":
        return rider
    return (min(rider[0], bike[0]), min(rider[1], bike[1]), max(rider[2], bike[2]), max(rider[3], bike[3]))


def candidate_pairs(riders, bikes, iou_min):
    """Все пары (iou, rider_id, bike_id) с IoU >= iou_min."""
    pairs = []
    for r in riders:
        for b in bikes:
            v = iou(box_of(r), box_of(b))
            if v >= iou_min:
                pairs.append((v, r["id"], b["id"]))
    return pairs


def match_global_greedy(pairs):
    """Пары по убыванию IoU; закрепляется пара, если rider и bike свободны."""
    used_r, used_b, result = set(), set(), {}
    for v, rid, bid in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        if rid in used_r or bid in used_b:
            continue
        used_r.add(rid)
        used_b.add(bid)
        result[rid] = (bid, v)
    return result


def match_best_only(pairs):
    """Каждый rider претендует только на свой лучший bike; если тот занят — rider отбрасывается."""
    best = {}
    for v, rid, bid in sorted(pairs, key=lambda p: (-p[0], p[1], p[2])):
        best.setdefault(rid, (bid, v))
    used_b, result = set(), {}
    for rid, (bid, v) in sorted(best.items(), key=lambda kv: (-kv[1][1], kv[0])):
        if bid in used_b:
            continue
        used_b.add(bid)
        result[rid] = (bid, v)
    return result


def ambiguous_rows(split, name, pairs, matched):
    """Спорные случаи: bike подходит нескольким rider; rider подходит нескольким bike."""
    by_bike, by_rider = {}, {}
    for v, rid, bid in pairs:
        by_bike.setdefault(bid, []).append((rid, v))
        by_rider.setdefault(rid, []).append((bid, v))
    assigned_bike = {rid: bid for rid, (bid, _) in matched.items()}
    rows = []
    for bid, lst in by_bike.items():
        if len(lst) > 1:
            winner = next((rid for rid, b in assigned_bike.items() if b == bid), "")
            rows.append({
                "split": split, "image": name, "case": "bike_multi_rider", "object_id": bid,
                "candidates": ";".join(f"{rid}:{v:.3f}" for rid, v in sorted(lst, key=lambda x: -x[1])),
                "assigned": winner,
            })
    for rid, lst in by_rider.items():
        if len(lst) > 1:
            ranked = sorted(lst, key=lambda x: -x[1])
            got = assigned_bike.get(rid, "")
            case = "rider_multi_bike"
            if got and got != ranked[0][0]:
                case = "rider_multi_bike_alt"  # лучший bike занят, взят свободный альтернативный
            elif not got:
                case = "rider_multi_bike_lost"
            rows.append({
                "split": split, "image": name, "case": case, "object_id": rid,
                "candidates": ";".join(f"{bid}:{v:.3f}" for bid, v in ranked),
                "assigned": got,
            })
    return rows


def occ_key(label):
    a = label.get("attributes", {})
    return (bool(a.get("occluded")), bool(a.get("truncated")))


def process_split(split, cfg, limit):
    path = Path(cfg["labels_dir"]) / cfg["label_file"].format(split=split)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]

    cat_r, cat_b = cfg["categories"]["rider"], cfg["categories"]["bike"]
    iou_min = cfg["matching"]["iou_min"]
    tod = cfg["filters"]["timeofday"]
    min_h = cfg["filters"]["min_height_px"]

    obj = Counter()
    imgs = {s: set() for s in STEPS}
    occ = {s: Counter() for s in STEPS[1:]}
    algo = Counter()
    ambiguous, selected = [], []

    for item in data:
        name = item["name"]
        imgs["all_images"].add(name)
        labels = item.get("labels") or []
        riders = [l for l in labels if l.get("category") == cat_r and l.get("box2d")]
        bikes = [l for l in labels if l.get("category") == cat_b and l.get("box2d")]
        if not riders:
            continue
        obj["rider"] += len(riders)
        imgs["rider"].add(name)
        for r in riders:
            occ["rider"][occ_key(r)] += 1

        pairs = candidate_pairs(riders, bikes, iou_min)
        matched = match_global_greedy(pairs)
        algo["nonexclusive"] += len({rid for _, rid, _ in pairs})
        algo["best_only"] += len(match_best_only(pairs))
        algo["global_greedy"] += len(matched)
        ambiguous += ambiguous_rows(split, name, pairs, matched)

        bikes_by_id = {b["id"]: b for b in bikes}
        is_day = item.get("attributes", {}).get("timeofday") == tod
        for r in riders:
            if r["id"] not in matched:
                continue
            bid, v = matched[r["id"]]
            b = bikes_by_id[bid]
            x1, y1, x2, y2 = box_of(r)
            bx1, by1, bx2, by2 = box_of(b)
            t = target_box((x1, y1, x2, y2), (bx1, by1, bx2, by2), cfg["target_box"])
            steps_passed = ["matched"]
            if is_day:
                steps_passed.append("daytime")
                if t[3] - t[1] >= min_h:
                    steps_passed.append("min_height")
            for s in steps_passed:
                obj[s] += 1
                imgs[s].add(name)
                occ[s][occ_key(r)] += 1
            ro, rt = occ_key(r)
            bo, bt = occ_key(b)
            ratio = area(t) / area((x1, y1, x2, y2)) if area((x1, y1, x2, y2)) else float("inf")
            if ratio > cfg["review"]["max_target_to_rider_area"]:
                ambiguous.append({"split": split, "image": name, "case": "target_much_larger_than_rider",
                                  "object_id": r["id"], "candidates": f"{bid}:{v:.3f}",
                                  "assigned": f"{bid} (площадь цели / rider = {ratio:.2f})"})
            selected.append({
                "image": name, "timeofday": item.get("attributes", {}).get("timeofday"),
                "rider_id": r["id"], "rider_x1": x1, "rider_y1": y1, "rider_x2": x2, "rider_y2": y2,
                "rider_occluded": ro, "rider_truncated": rt,
                "bike_id": bid, "bike_x1": bx1, "bike_y1": by1, "bike_x2": bx2, "bike_y2": by2,
                "bike_occluded": bo, "bike_truncated": bt,
                "target_x1": t[0], "target_y1": t[1], "target_x2": t[2], "target_y2": t[3],
                "target_to_rider_area": round(ratio, 3),
                "iou": round(v, 4), "last_step": steps_passed[-1],
            })

    counts = [{"split": split, "step": s, "images": len(imgs[s]), "objects": obj[s] if s != "all_images" else ""}
              for s in STEPS]
    occ_rows = [{"split": split, "step": s, "occluded": o, "truncated": t, "objects": occ[s][(o, t)]}
                for s in STEPS[1:] for o in (False, True) for t in (False, True)]
    algo_rows = [{"split": split, "algorithm": k, "matched_riders": algo[k]}
                 for k in ("nonexclusive", "best_only", "global_greedy")]
    return counts, occ_rows, algo_rows, ambiguous, selected


def write_csv(path, rows, fields=None):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summary_md(cfg, counts, algo_rows, ambiguous, occ_rows):
    lines = ["# BDD100K: отбор rider + bike", "",
             f"Сопоставление: `{cfg['matching']['algorithm']}`, IoU >= {cfg['matching']['iou_min']}; "
             f"timeofday = {cfg['filters']['timeofday']}; целевая рамка `{cfg['target_box']}`, "
             f"её высота >= {cfg['filters']['min_height_px']} px.", "",
             "## Счётчики после каждого шага", "",
             "| часть | шаг | изображений | объектов |", "|---|---|---|---|"]
    lines += [f"| {r['split']} | {r['step']} | {r['images']} | {r['objects']} |" for r in counts]

    lines += ["", "## Сравнение правил сопоставления (число сопоставленных rider)", "",
              "| часть | nonexclusive (введение) | best_only | global_greedy |", "|---|---|---|---|"]
    for split in cfg["splits"]:
        a = {r["algorithm"]: r["matched_riders"] for r in algo_rows if r["split"] == split}
        lines.append(f"| {split} | {a.get('nonexclusive', '')} | {a.get('best_only', '')} | {a.get('global_greedy', '')} |")

    lines += ["", "## Спорные случаи", "", "| часть | случай | число |", "|---|---|---|"]
    amb = Counter((r["split"], r["case"]) for r in ambiguous)
    lines += [f"| {s} | {c} | {n} |" for (s, c), n in sorted(amb.items())]

    ref = cfg.get("reference_counts", {})
    if ref:
        lines += ["", "## Сравнение с введением", "", "| часть | шаг | введение | сейчас | разница |", "|---|---|---|---|---|"]
        for split, steps in ref.items():
            now = {r["step"]: r["objects"] for r in counts if r["split"] == split}
            for step, val in steps.items():
                if step in now and now[step] != "":
                    lines.append(f"| {split} | {step} | {val} | {now[step]} | {now[step] - val:+d} |")

    lines += ["", "## occluded / truncated rider после последнего шага", "",
              "| часть | occluded | truncated | объектов |", "|---|---|---|---|"]
    lines += [f"| {r['split']} | {r['occluded']} | {r['truncated']} | {r['objects']} |"
              for r in occ_rows if r["step"] == "min_height"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bdd_filter.yaml")
    ap.add_argument("--limit", type=int, default=0, help="взять только первые N изображений каждой части")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    counts, occ_rows, algo_rows, ambiguous = [], [], [], []
    for split in cfg["splits"]:
        c, o, a, amb, sel = process_split(split, cfg, args.limit)
        counts += c
        occ_rows += o
        algo_rows += a
        ambiguous += amb
        write_csv(out / f"bdd_selected_{split}.csv", sel)
        for r in c:
            print(f"{split:5s} {r['step']:11s} images={r['images']:6d} objects={r['objects']}")

    write_csv(out / "bdd_counts.csv", counts)
    write_csv(out / "bdd_occ_trunc.csv", occ_rows)
    write_csv(out / "bdd_algorithms.csv", algo_rows)
    write_csv(out / "bdd_ambiguous.csv", ambiguous,
              ["split", "image", "case", "object_id", "candidates", "assigned"])
    (out / "bdd_filter.md").write_text(summary_md(cfg, counts, algo_rows, ambiguous, occ_rows), encoding="utf-8")
    print(f"Отчёты: {out}/bdd_*.csv, {out}/bdd_filter.md")


if __name__ == "__main__":
    main()
