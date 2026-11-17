"""Конвертация отобранных объектов KITTI и BDD100K в формат YOLO (класс Cyclist) + ignore-области.

Выход: см. configs/yolo.yaml; отчёты:
  reports/yolo_counts.csv, reports/yolo.md        изображения, цели, ignore по частям
  reports/yolo_examples.md                        по 5 пар «исходная запись -> YOLO» на датасет
  reports/yolo_box_shapes.md, examples/box_shapes.png  форма рамок KITTI Cyclist / BDD rider / BDD rider∪bike
  reports/examples/{kitti,bdd}_yolo_*.png         кадры с целями (зелёные) и ignore (жёлтые)

Запуск: python src/to_yolo.py [--config configs/yolo.yaml] [--dataset kitti|bdd] [--limit N]
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

PARTS = ["train", "calib", "test"]


def yolo_line(box, W, H):
    x1, y1, x2, y2 = max(0.0, box[0]), max(0.0, box[1]), min(float(W), box[2]), min(float(H), box[3])
    return f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


def link(src, dst):
    """Жёсткая ссылка (тот же диск, без копирования); если нельзя — копия."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def read_split(cfg, ds, part):
    return sorted((Path(cfg["splits_dir"]) / f"{ds}_{part}.txt").read_text(encoding="utf-8").split())


# ---------------------------------------------------------------- KITTI

def kitti_records(cfg, limit):
    """{part: [(image_file, W, H, targets[(box, raw)], ignores[(box, reason, raw)])]}"""
    from PIL import Image

    c = cfg["kitti"]
    selected = set()
    with open(c["selected_file"], encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            selected.add((r["image"], int(r["line"])))
    out = {}
    for part in PARTS:
        recs = []
        for stem in read_split(cfg, "kitti", part)[: limit or None]:
            img = Path(c["image_dir"]) / f"{stem}{c['image_ext']}"
            with Image.open(img) as im:
                W, H = im.size
            tg, ig = [], []
            for i, ln in enumerate((Path(c["label_dir"]) / f"{stem}.txt").read_text().splitlines()):
                p = ln.split()
                if not p:
                    continue
                box = tuple(map(float, p[4:8]))
                if p[0] == c["target"] and (stem, i) in selected:
                    tg.append((box, ln))
                elif p[0] in c["ignore_categories"]:
                    reason = "dontcare" if p[0] == "DontCare" else f"cyclist_filtered_occ{p[2]}_tr{float(p[1]):.2f}"
                    ig.append((box, reason, ln))
            recs.append((img, W, H, tg, ig))
        out[part] = recs
    return out


# ---------------------------------------------------------------- BDD100K

def bdd_box(lb):
    b = lb["box2d"]
    return (b["x1"], b["y1"], b["x2"], b["y2"])


def bdd_records(cfg, limit):
    c = cfg["bdd"]
    W, H = c["image_size"]
    out, cache = {}, {}
    for part in PARTS:
        src = c["sources"][part]
        if src not in cache:
            with open(Path(c["labels_dir"]) / c["label_file"].format(source=src), encoding="utf-8") as fh:
                cache[src] = {it["name"]: it for it in json.load(fh)}
        items = cache[src]
        # Целевая рамка берётся из результата src/bdd_filter.py (там же задан её тип: rider или union).
        sel, matched_box = {}, {}
        with open(c["selected_file"].format(source=src), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                box = tuple(float(r[f"target_{k}"]) for k in ("x1", "y1", "x2", "y2"))
                matched_box[r["rider_id"]] = box
                if r["last_step"] == c["final_step"]:
                    sel[r["rider_id"]] = (r["bike_id"], box)
        recs, dropped = [], 0
        for name in read_split(cfg, "bdd", part):
            it = items[name]
            if it.get("attributes", {}).get("timeofday") != c["timeofday"]:
                dropped += 1
                continue
            if limit and len(recs) >= limit:
                break
            labels = [lb for lb in it.get("labels") or [] if lb.get("box2d")]
            by_id = {str(lb["id"]): lb for lb in labels}
            tg, ig = [], []
            for lb in labels:
                lid = str(lb["id"])
                if lb["category"] == "rider" and lid in sel:
                    bike_id, box = sel[lid]
                    tg.append((box, {"rider": lb, "bike": by_id.get(str(bike_id))}))
                elif lb["category"] in c["ignore_categories"]:
                    # rider с велосипедом, отсеянный порогом высоты, — ignore по его целевой рамке
                    ig.append((matched_box.get(lid, bdd_box(lb)), f"{lb['category']}_filtered", lb))
            img = Path(c["image_dir"].format(source=src)) / name
            recs.append((img, W, H, tg, ig))
        out[part] = recs
        out[f"_{part}_dropped"] = dropped
    return out


# ---------------------------------------------------------------- запись

def write_dataset(cfg, ds, recs):
    root = Path(cfg["out_dir"]) / ds
    counts = []
    for part in PARTS:
        for sub in ("images", "labels", "ignore"):
            (root / sub / part).mkdir(parents=True, exist_ok=True)
        n_img = n_tg = n_ig = n_missing = n_pos = 0
        for img, W, H, tg, ig in recs[part]:
            stem = Path(img).stem
            (root / "labels" / part / f"{stem}.txt").write_text(
                "".join(yolo_line(b, W, H) + "\n" for b, _ in tg), encoding="utf-8")
            (root / "ignore" / part / f"{stem}.txt").write_text(
                "".join(f"{b[0]:.2f} {b[1]:.2f} {b[2]:.2f} {b[3]:.2f} {reason}\n" for b, reason, _ in ig),
                encoding="utf-8")
            if Path(img).exists():
                link(Path(img), root / "images" / part / Path(img).name)
            else:
                n_missing += 1
            n_img += 1
            n_tg += len(tg)
            n_ig += len(ig)
            n_pos += bool(tg)
        counts.append({"dataset": ds, "part": part, "images": n_img, "images_with_targets": n_pos,
                       "targets": n_tg, "ignore_regions": n_ig, "images_missing_on_disk": n_missing,
                       "dropped_not_" + ("daytime" if ds == "bdd" else "na"): recs.get(f"_{part}_dropped", 0)})
    (Path(cfg["out_dir"]) / f"{ds}.yaml").write_text(yaml.safe_dump({
        "path": str(root.resolve()), "train": "images/train", "val": "images/calib", "test": "images/test",
        "names": {0: cfg["class_name"]},
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return counts


# ---------------------------------------------------------------- примеры и форма рамок

def examples_md(cfg, ds, recs):
    rng = np.random.default_rng(0)
    pool = [(r, t) for r in recs["test"] + recs["train"] for t in r[3]]
    picks = [pool[i] for i in sorted(rng.choice(len(pool), size=min(cfg["n_examples"], len(pool)), replace=False))]
    lines = [f"## {ds}", ""]
    for (img, W, H, _, _), (box, raw) in picks:
        lines += [f"**{Path(img).name}** ({W}×{H})", "", "Исходная запись:", "```"]
        if ds == "kitti":
            lines.append(raw)
        else:
            for k in ("rider", "bike"):
                lb = raw[k]
                if lb:
                    lines.append(json.dumps({"id": lb["id"], "category": lb["category"], "box2d": lb["box2d"],
                                             "attributes": {a: lb["attributes"].get(a) for a in ("occluded", "truncated")}},
                                            ensure_ascii=False))
        lines += ["```", "YOLO:", "```", yolo_line(box, W, H), "```", ""]
    return lines


def box_shapes(cfg):
    """Отношение высоты к ширине: KITTI Cyclist против BDD rider и BDD rider∪bike (отобранные объекты)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = []
    with open(cfg["kitti"]["selected_file"], encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k.append((float(r["bottom"]) - float(r["top"])) / (float(r["right"]) - float(r["left"])))
    rider, union, bike_out = [], [], []
    for src in ("train", "val"):
        with open(cfg["bdd"]["selected_file"].format(source=src), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["last_step"] != cfg["bdd"]["final_step"]:
                    continue
                rb = [float(r[f"rider_{a}"]) for a in ("x1", "y1", "x2", "y2")]
                bb = [float(r[f"bike_{a}"]) for a in ("x1", "y1", "x2", "y2")]
                ub = [min(rb[0], bb[0]), min(rb[1], bb[1]), max(rb[2], bb[2]), max(rb[3], bb[3])]
                rider.append((rb[3] - rb[1]) / (rb[2] - rb[0]))
                union.append((ub[3] - ub[1]) / (ub[2] - ub[0]))
                ua = (ub[2] - ub[0]) * (ub[3] - ub[1])
                bike_out.append(1 - (rb[2] - rb[0]) * (rb[3] - rb[1]) / ua)
    sets = {"KITTI Cyclist": k, "BDD rider": rider, "BDD rider∪bike": union}
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 4, 41)
    for name, v in sets.items():
        ax.hist(v, bins=bins, histtype="step", density=True, linewidth=1.8, label=f"{name} (n={len(v)})")
    ax.set_xlabel("высота / ширина рамки")
    ax.set_ylabel("плотность")
    ax.legend()
    fig.tight_layout()
    out = Path(cfg["examples_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "box_shapes.png", dpi=120)
    plt.close(fig)
    q = lambda v: " / ".join(f"{x:.2f}" for x in np.quantile(v, [0.25, 0.5, 0.75]))
    lines = ["# Форма целевой рамки: KITTI Cyclist против BDD100K", "",
             "Отобранные объекты (после всех фильтров). Отношение высота/ширина, квартили Q1 / медиана / Q3.", "",
             "| рамка | объектов | высота/ширина |", "|---|---|---|"]
    lines += [f"| {n} | {len(v)} | {q(v)} |" for n, v in sets.items()]
    lines += ["", f"Доля площади rider∪bike вне рамки rider (медиана): {np.median(bike_out):.0%}. "
              "KITTI Cyclist размечает человека вместе с велосипедом, BDD rider — только человека.", "",
              "![](examples/box_shapes.png)"]
    (Path(cfg["report_dir"]) / "yolo_box_shapes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[4:8]))


def draw_examples(cfg, ds, recs, n=4):
    import cv2

    cand = [r for r in recs["test"] if r[3] and Path(r[0]).exists()]
    if not cand:
        print(f"{ds}: нет изображений на диске — картинки с рамками не построены")
        return
    cand.sort(key=lambda r: (-(len(r[3]) + len(r[4])), str(r[0])))
    out = Path(cfg["examples_dir"])
    for i, (img, W, H, tg, ig) in enumerate(cand[:n]):
        im = cv2.imread(str(img))
        for b, reason, _ in ig:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 255), 1)
        for b, raw in tg:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            if ds == "bdd" and raw.get("bike"):
                bb = bdd_box(raw["bike"])
                cv2.rectangle(im, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 128, 0), 1)
        cv2.imwrite(str(out / f"{ds}_yolo_{i}.png"), im)
    print(f"{ds}: {min(n, len(cand))} картинок -> {out}/{ds}_yolo_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/yolo.yaml")
    ap.add_argument("--dataset", choices=["kitti", "bdd"], action="append")
    ap.add_argument("--limit", type=int, default=0, help="только первые N изображений каждой части (smoke test)")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    rep = Path(cfg["report_dir"])
    counts, ex = [], ["# Примеры конвертации: исходная запись -> YOLO", "",
                      "Формат YOLO: `класс cx cy w h`, координаты центра и размеры нормированы на ширину и высоту кадра.", ""]
    for ds in args.dataset or ["kitti", "bdd"]:
        recs = kitti_records(cfg, args.limit) if ds == "kitti" else bdd_records(cfg, args.limit)
        c = write_dataset(cfg, ds, recs)
        counts += c
        ex += examples_md(cfg, ds, recs)
        draw_examples(cfg, ds, recs)
        for r in c:
            print(" ".join(f"{k}={v}" for k, v in r.items()))
    (rep / "yolo_examples.md").write_text("\n".join(ex) + "\n", encoding="utf-8")
    keys = ["dataset", "part", "images", "images_with_targets", "targets", "ignore_regions", "images_missing_on_disk"]
    with open(rep / "yolo_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys + ["dropped"])
        w.writeheader()
        for r in counts:
            w.writerow({**{k: r[k] for k in keys}, "dropped": next((v for k, v in r.items() if k.startswith("dropped")), 0)})
    with open(cfg["bdd"]["filter_config"], encoding="utf-8") as fh:
        bdd_target = yaml.safe_load(fh)["target_box"]
    md = ["# Конвертация в YOLO", "", f"Класс: {cfg['class_name']} (id 0). BDD: целевая рамка `{bdd_target}`, "
          f"только кадры timeofday = {cfg['bdd']['timeofday']}.", "",
          "| датасет | часть | изображений | с целями | целей | ignore | нет файла изображения | отброшено (не daytime) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in counts:
        d = next((v for k, v in r.items() if k.startswith("dropped")), 0)
        md.append(f"| {r['dataset']} | {r['part']} | {r['images']} | {r['images_with_targets']} | {r['targets']} | "
                  f"{r['ignore_regions']} | {r['images_missing_on_disk']} | {d if r['dataset'] == 'bdd' else '—'} |")
    (rep / "yolo.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    box_shapes(cfg)


if __name__ == "__main__":
    main()
