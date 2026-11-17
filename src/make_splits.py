"""Разбиение BDD100K и KITTI на непересекающиеся train / calib / test по кластерам.

BDD100K: test = официальный val; calib и train — из официального train, по кластерам-префиксам имени.
KITTI:   все три части — из training, по поездкам KITTI raw (нужен mapping из devkit_object).

Выход:
  splits/{bdd,kitti}_{train,calib,test}.txt   имена изображений (по одному в строке)
  splits/{bdd,kitti}_clusters.csv             изображение -> кластер -> часть
  reports/splits.md, reports/splits_counts.csv  размеры частей и проверка пересечений

Запуск: python src/make_splits.py [--config configs/splits.yaml] [--only bdd|kitti] [--check-only]
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PARTS = ["train", "calib", "test"]


def bdd_cluster(name):
    return name.split("-")[0]


def selected_counts(path, image_key, final_step=None):
    """Число отобранных объектов (и из них исходно неперекрытых) на изображение."""
    total, visible = Counter(), Counter()
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if final_step and row["last_step"] != final_step:
                continue
            img = image_key(row)
            total[img] += 1
            occ = row.get("rider_occluded", row.get("occluded"))
            if occ in ("False", "0"):
                visible[img] += 1
    return total, visible


def fill_by_share(clusters, objects_in_cluster, targets, seed):
    """Перемешать кластеры и заполнять части по очереди до целевой доли объектов.

    targets — список (часть, доля); оставшиеся кластеры уходят в train.
    """
    order = sorted(clusters)
    random.Random(seed).shuffle(order)
    total = sum(objects_in_cluster[c] for c in order) or 1
    assign, i = {}, 0
    for part, share in targets:
        got = 0
        while i < len(order) and got / total < share:
            assign[order[i]] = part
            got += objects_in_cluster[order[i]]
            i += 1
    for c in order[i:]:
        assign[c] = "train"
    return assign


def prefix_evidence(items, seed):
    """Доля групп-префиксов (размер > 1) с одинаковым атрибутом против случайных групп того же размера."""
    groups = defaultdict(list)
    for it in items:
        groups[bdd_cluster(it["name"])].append(it.get("attributes", {}))
    real = [g for g in groups.values() if len(g) > 1]
    pool = [a for g in real for a in g]
    random.Random(seed).shuffle(pool)
    rnd, k = [], 0
    for g in real:
        rnd.append(pool[k:k + len(g)])
        k += len(g)
    same = lambda gs, a: sum(len({x.get(a) for x in g}) == 1 for g in gs) / len(gs)
    return len(real), {a: (same(real, a), same(rnd, a)) for a in ("timeofday", "weather", "scene")}


def make_bdd(cfg, seed):
    c = cfg["bdd"]
    names, notes = {}, []
    for src in (c["train_source"], c["test_source"]):
        with open(Path(c["labels_dir"]) / c["label_file"].format(split=src), encoding="utf-8") as fh:
            items = json.load(fh)
        names[src] = [it["name"] for it in items]
        if len(set(names[src])) != len(names[src]):
            notes.append(f"BDD {src}: имена файлов не уникальны.")
        n_groups, ev = prefix_evidence(items, seed)
        notes.append(f"BDD {src}: {len(names[src])} уникальных имён, {len({bdd_cluster(n) for n in names[src]})} префиксов, "
                     f"{n_groups} префиксов с > 1 изображением. Доля таких групп с одинаковым атрибутом "
                     "(реальные группы / случайные группы того же размера): "
                     + "; ".join(f"{a} {r:.2f} / {x:.2f}" for a, (r, x) in ev.items()) + ".")

    obj, vis = Counter(), Counter()
    for src in names:
        t, v = selected_counts(c["selected_file"].format(split=src), lambda r: r["image"], c["final_step"])
        obj.update(t)
        vis.update(v)

    test_clusters = {bdd_cluster(n) for n in names[c["test_source"]]}
    rows = []
    for n in names[c["test_source"]]:
        rows.append((n, bdd_cluster(n), "test"))

    train_names = names[c["train_source"]]
    leaked = [n for n in train_names if bdd_cluster(n) in test_clusters]
    if leaked:
        notes.append(f"Из official train исключено {len(leaked)} изображений: их кластер есть в official val "
                     f"({', '.join(sorted({bdd_cluster(n) for n in leaked}))}).")
    train_names = [n for n in train_names if bdd_cluster(n) not in test_clusters]

    cl_obj = Counter()
    for n in train_names:
        cl_obj[bdd_cluster(n)] += obj[n]
    assign = fill_by_share(set(map(bdd_cluster, train_names)), cl_obj, [("calib", c["calib_object_share"])], seed)
    rows += [(n, bdd_cluster(n), assign[bdd_cluster(n)]) for n in train_names]
    return rows, obj, vis, notes


def read_kitti_mapping(c):
    mp, rp = Path(c["mapping_file"]), Path(c["rand_file"])
    if not (mp.exists() and rp.exists()):
        return None
    mapping = [ln.split() for ln in mp.read_text().splitlines() if ln.strip()]
    rand = [int(x) for x in rp.read_text().replace("\n", "").split(",") if x.strip()]
    # rand[i] — номер (с 1) строки mapping для кадра training/i; строка: дата, поездка, кадр raw.
    return {f"{i:06d}": mapping[r - 1][1] for i, r in enumerate(rand)}


def make_kitti(cfg, seed):
    c = cfg["kitti"]
    drive_of = read_kitti_mapping(c)
    if drive_of is None:
        return None
    images = sorted(p.stem for p in Path(c["label_dir"]).glob("*.txt"))
    missing = [i for i in images if i not in drive_of]
    if missing:
        sys.exit(f"В mapping нет {len(missing)} кадров KITTI, например {missing[:3]}")
    obj, vis = selected_counts(c["selected_file"], lambda r: r["image"])
    cl_obj = Counter()
    for i in images:
        cl_obj[drive_of[i]] += obj[i]
    assign = fill_by_share(set(drive_of[i] for i in images), cl_obj,
                           [("test", c["test_object_share"]), ("calib", c["calib_object_share"])], seed)
    rows = [(i, drive_of[i], assign[drive_of[i]]) for i in images]
    return rows, obj, vis, []


def save(ds, rows, out):
    out.mkdir(parents=True, exist_ok=True)
    by_part = defaultdict(list)
    for img, cl, part in rows:
        by_part[part].append(img)
    for part in PARTS:
        (out / f"{ds}_{part}.txt").write_text("\n".join(sorted(by_part[part])) + "\n", encoding="utf-8")
    with open(out / f"{ds}_clusters.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "cluster", "part"])
        w.writerows(sorted(rows))


def check(ds, out):
    """Пересечения изображений и кластеров между частями — по сохранённым файлам."""
    imgs = {p: set((out / f"{ds}_{p}.txt").read_text(encoding="utf-8").split()) for p in PARTS}
    cl = defaultdict(set)
    with open(out / f"{ds}_clusters.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cl[r["part"]].add(r["cluster"])
    res = []
    for a, b in (("train", "calib"), ("train", "test"), ("calib", "test")):
        res.append((ds, f"{a}∩{b}", len(imgs[a] & imgs[b]), len(cl[a] & cl[b])))
    return res


def stats(ds, rows, obj, vis):
    res = []
    for part in PARTS:
        sel = [r for r in rows if r[2] == part]
        per_cluster = Counter()
        for r in sel:
            per_cluster[r[1]] += obj[r[0]]
        res.append({
            "dataset": ds, "part": part, "images": len(sel), "clusters": len({r[1] for r in sel}),
            "clusters_with_objects": sum(1 for v in per_cluster.values() if v),
            "max_objects_in_cluster": max(per_cluster.values(), default=0),
            "images_with_objects": sum(1 for r in sel if obj[r[0]]),
            "objects": sum(obj[r[0]] for r in sel), "objects_not_occluded": sum(vis[r[0]] for r in sel),
        })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/splits.yaml")
    ap.add_argument("--only", choices=["bdd", "kitti"])
    ap.add_argument("--check-only", action="store_true", help="только проверить пересечения сохранённых списков")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    out, rep = Path(cfg["out_dir"]), Path(cfg["report_dir"])
    datasets = [args.only] if args.only else ["bdd", "kitti"]

    counts, checks, notes = [], [], []
    for ds in datasets:
        if args.check_only:
            if (out / f"{ds}_clusters.csv").exists():
                checks += check(ds, out)
            continue
        made = make_bdd(cfg, cfg["seed"]) if ds == "bdd" else make_kitti(cfg, cfg["seed"])
        if made is None:
            notes.append(f"{ds}: нет файлов mapping ({cfg['kitti']['mapping_file']}, {cfg['kitti']['rand_file']}) — "
                         "разбиение не построено.")
            print(notes[-1])
            continue
        rows, obj, vis, n = made
        notes += n
        save(ds, rows, out)
        counts += stats(ds, rows, obj, vis)
        checks += check(ds, out)

    for ds, pair, ni, nc in checks:
        print(f"{ds:5s} {pair:12s} общих изображений={ni} общих кластеров={nc}")
    if args.check_only:
        sys.exit(1 if any(ni or nc for _, _, ni, nc in checks) else 0)

    if counts:
        with open(rep / "splits_counts.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(counts[0].keys()))
            w.writeheader()
            w.writerows(counts)
    for r in counts:
        print(f"{r['dataset']:5s} {r['part']:5s} images={r['images']:6d} clusters={r['clusters']:6d} "
              f"objects={r['objects']:5d} not_occluded={r['objects_not_occluded']}")

    lines = ["# Разбиение train / calib / test", "", f"seed = {cfg['seed']}. Части собираются из целых кластеров.", "",
             "| датасет | часть | изображений | кластеров | кластеров с объектами | макс. объектов в кластере "
             "| изображений с объектами | объектов | из них не перекрыты |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dataset']} | {r['part']} | {r['images']} | {r['clusters']} | {r['clusters_with_objects']} | "
              f"{r['max_objects_in_cluster']} | {r['images_with_objects']} | {r['objects']} | {r['objects_not_occluded']} |"
              for r in counts]
    lines += ["", "## Проверка пересечений", "", "| датасет | пара | общих изображений | общих кластеров |", "|---|---|---|---|"]
    lines += [f"| {ds} | {pair} | {ni} | {nc} |" for ds, pair, ni, nc in checks]
    if notes:
        lines += ["", "## Примечания", ""] + [f"- {n}" for n in notes]
    (rep / "splits.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if any(ni or nc for _, _, ni, nc in checks):
        sys.exit("Найдены пересечения между частями")


if __name__ == "__main__":
    main()
