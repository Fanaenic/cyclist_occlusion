"""Весь конвейер: обучение -> порог на calib -> оценка test на всех уровнях масок -> бутстрэп.

Для каждого датасета, режима (baseline / proposed / ORE) и seed:
  1. обучение YOLOv8n (src/occlusion_train.py — маски proposed / ORE при обучении);
  2. детекции на calib без масок -> порог для FPPI = target_fppi (заморожен);
  3. детекции на test для каждого уровня маски (маски из splits/masks_*_test.csv, одни и те же для всех моделей)
     -> src/evaluate.py: метрики и .npz по изображениям;
  4. src/stats.py по всем .npz.

Выход: {out_dir}/{tag}/ — веса, детекции, eval/*.npz, eval/thresholds.json, metrics.csv;
       reports/stats_{tag}.*, reports/{tag}.md

Запуск:
  python src/run_experiment.py --smoke                     технический прогон (limit и epochs из smoke)
  python src/run_experiment.py --tag main                  полная серия — только по явной команде
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import SUBSETS, evaluate, level_ground_truth, load_gt  # noqa: E402
from masks import apply_mask  # noqa: E402

PARTS = ["train", "calib", "test"]


def load(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------- данные

def subset_images(root, part, limit):
    """Первые limit изображений части: сначала с целями, затем без (по имени)."""
    labels = sorted((root / "labels" / part).glob("*.txt"))
    pos = [p.stem for p in labels if p.stat().st_size > 0]
    neg = [p.stem for p in labels if p.stat().st_size == 0]
    return (pos + neg)[:limit] if limit else pos + neg


def build_subset(ycfg, ds, limit, out):
    """Набор для Ultralytics: при limit — подмножество (жёсткие ссылки), иначе — datasets_yolo как есть."""
    root = Path(ycfg["out_dir"]) / ds
    images = {p: subset_images(root, p, limit) for p in PARTS}
    if not limit:
        return Path(ycfg["out_dir"]) / f"{ds}.yaml", images, root
    sub = out / "data" / ds
    for p in PARTS:
        for d in ("images", "labels"):
            (sub / d / p).mkdir(parents=True, exist_ok=True)
        files = {f.stem: f for f in (root / "images" / p).iterdir()}
        for stem in images[p]:
            src = files[stem]
            dst = sub / "images" / p / src.name
            if not dst.exists():
                os.link(src, dst)
            (sub / "labels" / p / f"{stem}.txt").write_bytes((root / "labels" / p / f"{stem}.txt").read_bytes())
    data_yaml = out / "data" / f"{ds}.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(sub.resolve()), "train": "images/train", "val": "images/calib",
                                         "test": "images/test", "names": {0: ycfg["class_name"]}}), encoding="utf-8")
    return data_yaml, images, root


# ---------------------------------------------------------------- обучение

def train(tcfg, mcfg, data_yaml, ds, regime, seed, epochs, out):
    from ultralytics import YOLO
    from occlusion_train import OcclusionTrainer

    name = f"{ds}__{regime}__s{seed}"
    weights = out / "train" / name / "weights" / "last.pt"      # модель — веса последней эпохи (см. train.yaml)
    done = out / "train" / name / "done.json"                   # метка завершённого обучения
    if done.exists():
        return weights
    OcclusionTrainer.occ = {"regime": regime, "cfg": mcfg, "seed": mcfg["training"]["seed"] * 1000 + seed}
    model = YOLO(tcfg["model"])
    model.train(trainer=OcclusionTrainer, data=str(data_yaml), epochs=epochs, imgsz=tcfg["imgsz"],
                batch=tcfg["batch"], seed=seed, deterministic=tcfg["deterministic"], workers=tcfg["workers"],
                amp=tcfg["amp"], plots=tcfg["plots"], val=tcfg["val"], project=str((out / "train").resolve()),
                name=name, exist_ok=True, verbose=False)
    done.write_text(json.dumps({"epochs": epochs, "seed": seed, "regime": regime}), encoding="utf-8")
    return weights


# ---------------------------------------------------------------- инференс

def detect(weights, image_paths, masks_by_image, tcfg, pcfg, fill):
    """Детекции на изображениях (с наложенными масками уровня). Возвращает {stem: [(x1,y1,x2,y2,score)]}."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    out = {}
    bs = tcfg["inference"]["batch"]
    for i in range(0, len(image_paths), bs):
        chunk = image_paths[i:i + bs]
        imgs = []
        for p in chunk:
            orig = cv2.imread(str(p))
            im = orig
            for m, seed, src in masks_by_image.get(p.stem, []):
                im = apply_mask(im, m, fill, seed, src, orig)     # заливка фоном берётся из исходного кадра
            imgs.append(im)
        res = model.predict(imgs, imgsz=tcfg["imgsz"], conf=pcfg["nms"]["conf"], iou=pcfg["nms"]["iou"],
                            max_det=tcfg["inference"]["max_det_raw"], verbose=False)
        for p, r in zip(chunk, res):
            b = r.boxes
            out[p.stem] = [(*xy, float(s)) for xy, s in zip(b.xyxy.cpu().numpy().tolist(), b.conf.cpu().numpy())]
    return out


def read_masks(splits_dir, ds, level):
    """{stem: [((x0,y0,x1,y1), mask_seed, (sx, sy))]} для уровня (уровень 0 — пусто)."""
    res = {}
    if level == 0:
        return res
    with open(Path(splits_dir) / f"masks_{ds}_test.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if float(r["level"]) == level:
                res.setdefault(Path(r["image"]).stem, []).append(
                    (tuple(int(r[k]) for k in ("mx0", "my0", "mx1", "my1")), int(r["mask_seed"]),
                     (r["sx"], r["sy"])))
    return res


def read_control(splits_dir, ds):
    """Контрольные заплатки на фоне: {stem: [((x0,y0,x1,y1), mask_seed, (sx, sy))]}."""
    res = {}
    with open(Path(splits_dir) / f"control_{ds}_test.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            res.setdefault(Path(r["image"]).stem, []).append(
                (tuple(int(r[k]) for k in ("dx0", "dy0", "dx1", "dy1")), int(r["mask_seed"]), (r["sx"], r["sy"])))
    return res


def control_check(dets, control, thr, pcfg):
    """Доля контрольных заплаток, на которых есть детекция (после NMS, score >= thr), и число таких детекций.

    Детекция считается «на заплатке», если IoU с ней >= matching.iou или >= ignore_ioa площади детекции внутри неё.
    """
    from evaluate import ioa, iou, nms
    hit, n, n_det = 0, 0, 0
    for img, patches in control.items():
        ds_ = [d for d in nms(dets.get(img, []), pcfg) if d[4] >= thr]
        for m, _, _ in patches:
            n += 1
            on = [d for d in ds_ if iou(d, m) >= pcfg["matching"]["iou"] or ioa(d, m) >= pcfg["matching"]["ignore_ioa"]]
            hit += bool(on)
            n_det += len(on)
    return {"patches": n, "patches_hit": hit, "hit_rate": hit / n if n else 0.0, "detections_on_patches": n_det}


def mask_objects(splits_dir, ds, level):
    """{stem: {object_id: mask}} — для перевода целей в ignore по маскам соседей."""
    res = {}
    if level == 0:
        return res
    with open(Path(splits_dir) / f"masks_{ds}_test.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if float(r["level"]) == level:
                res.setdefault(Path(r["image"]).stem, {})[r["object_id"]] = \
                    tuple(int(r[k]) for k in ("mx0", "my0", "mx1", "my1"))
    return res


def evaluate_part(dets, ds, part, images, level, subset, pcfg, ycfg, splits_dir):
    _, targets, ignores, _ = load_gt(ds, part, ycfg)
    mobj = mask_objects(splits_dir, ds, level) if part == "test" else {}
    gt = {}
    for img in images:
        keep, ign, _ = level_ground_truth(targets.get(img, []), ignores.get(img, []), mobj.get(img, {}),
                                          pcfg, SUBSETS[subset])
        gt[img] = (keep, ign)
    ev, _ = evaluate(dets, gt, images, pcfg)
    return ev


# ---------------------------------------------------------------- главный цикл

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag")
    ap.add_argument("--dataset", choices=["kitti", "bdd"], action="append")
    ap.add_argument("--regimes", nargs="+")
    ap.add_argument("--replicas", type=int, help="переопределить число реплик бутстрэпа")
    args = ap.parse_args()
    tcfg = load(args.train_config)
    mcfg, pcfg, ycfg = load("configs/masks.yaml"), load("configs/protocol.yaml"), load("configs/yolo.yaml")
    if tcfg.get("offline"):
        os.environ["YOLO_OFFLINE"] = "1"
    tag = args.tag or ("smoke" if args.smoke else None)
    if not tag:
        sys.exit("Полная серия запускается только с явным --tag (см. CLAUDE.md)")
    limit = tcfg["smoke"]["limit"] if args.smoke else 0
    epochs = tcfg["smoke"]["epochs"] if args.smoke else tcfg["epochs"]
    out = Path(tcfg["out_dir"]) / tag
    ev_dir = out / "eval"
    ev_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.dataset or tcfg["datasets"]
    regimes = args.regimes or tcfg["regimes"]
    # при запуске части датасетов имена отчётов включают их, чтобы не затирать результаты других датасетов
    rtag = tag if set(datasets) == set(tcfg["datasets"]) else f"{tag}_{'_'.join(datasets)}"
    levels = mcfg["levels"]
    subsets = ["all", "not_occluded"]
    thr_path = ev_dir / "thresholds.json"
    thresholds = json.loads(thr_path.read_text()) if thr_path.exists() else {}
    metrics_rows, control_rows, timing = [], [], {}

    for ds in datasets:
        data_yaml, images, root = build_subset(ycfg, ds, limit, out)
        img_files = {p: {f.stem: f for f in (root / "images" / p).iterdir()} for p in ("calib", "test")}
        print(f"[{ds}] train/calib/test = {[len(images[p]) for p in PARTS]} изображений")
        for regime in regimes:
            for seed in tcfg["seeds"]:
                key = f"{ds}__{regime}__s{seed}"
                t0 = time.time()
                w = train(tcfg, mcfg, data_yaml, ds, regime, seed, epochs, out)
                timing[f"{key} train"] = time.time() - t0
                t0 = time.time()
                calib_paths = [img_files["calib"][s] for s in images["calib"]]
                ev = evaluate_part(detect(w, calib_paths, {}, tcfg, pcfg, mcfg["fill"]), ds, "calib",
                                   images["calib"], 0.0, "all", pcfg, ycfg, mcfg["splits_dir"])
                thresholds[key] = ev.threshold_for_fppi(pcfg["metrics"]["target_fppi"])
                thr_path.write_text(json.dumps(thresholds, indent=1))
                test_paths = [img_files["test"][s] for s in images["test"]]
                for level in levels:
                    dets = detect(w, test_paths, read_masks(mcfg["splits_dir"], ds, level), tcfg, pcfg, mcfg["fill"])
                    for subset in subsets:
                        if subset == "not_occluded" and ds != "bdd":
                            continue
                        ev = evaluate_part(dets, ds, "test", images["test"], level, subset, pcfg, ycfg,
                                           mcfg["splits_dir"])
                        ev.save(ev_dir / f"{key}__L{level}__{subset}.npz", dataset=ds, regime=regime, seed=seed,
                                level=level, subset=subset, threshold=thresholds[key])
                        at = ev.at_threshold(thresholds[key])
                        metrics_rows.append({"dataset": ds, "regime": regime, "seed": seed, "level": level,
                                             "subset": subset, "threshold": thresholds[key], "targets": ev.n_targets,
                                             "images": ev.n_images, **at, "ap50": ev.ap(),
                                             "recall_at_fppi_curve": ev.recall_at_fppi(pcfg["metrics"]["target_fppi"])})
                        print(f"  {key} L={level:.0%} {subset:12s} R={at['recall']:.3f} FPPI={at['fppi']:.3f} "
                              f"AP={ev.ap():.3f} thr={thresholds[key]:.3f}")
                # контроль: заплатки только на фоне, цели открыты
                ctrl = {s: v for s, v in read_control(mcfg["splits_dir"], ds).items() if s in set(images["test"])}
                cpaths = [img_files["test"][s] for s in sorted(ctrl)]
                cdets = detect(w, cpaths, ctrl, tcfg, pcfg, mcfg["fill"])
                cres = control_check(cdets, ctrl, thresholds[key], pcfg)
                ev0 = evaluate_part(cdets, ds, "test", sorted(ctrl), 0.0, "all", pcfg, ycfg, mcfg["splits_dir"])
                cres.update(dataset=ds, regime=regime, seed=seed, fppi=ev0.at_threshold(thresholds[key])["fppi"],
                            recall=ev0.at_threshold(thresholds[key])["recall"])
                control_rows.append(cres)
                print(f"  {key} control: заплаток {cres['patches']}, с детекцией {cres['patches_hit']} "
                      f"({cres['hit_rate']:.3f}); FPPI={cres['fppi']:.3f}")
                timing[f"{key} eval"] = time.time() - t0

    with open(out / f"metrics_{rtag}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(metrics_rows[0].keys()))
        wr.writeheader()
        wr.writerows(metrics_rows)
    (out / f"timing_{rtag}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")

    from stats import analyse
    override = {"datasets": datasets}
    if args.replicas:
        override["bootstrap"] = {**pcfg["stats"]["bootstrap"], "replicas": args.replicas}
    t0 = time.time()
    _, decision = analyse(ev_dir, pcfg, override, tag=rtag)
    if "ore" in regimes:
        analyse(ev_dir, pcfg, override, tag=f"{rtag}_vs_ore", control="ore")
    if any(ds == "bdd" for ds in datasets):
        analyse(ev_dir, pcfg, {**override, "datasets": ["bdd"], "subset": "not_occluded"},
                tag=f"{rtag}_not_occluded")
    timing["stats"] = time.time() - t0
    (out / f"timing_{rtag}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    decision = apply_control_rule(control_rows, decision, pcfg, rtag, out)
    print(json.dumps(decision, ensure_ascii=False))


def apply_control_rule(rows, decision, pcfg, rtag, out):
    """Правило протокола (control): если доля контрольных заплаток с детекцией у treatment выше, чем у control,
    больше чем на max_hit_gap, вывод о полезности не делается (возможное обучение на заплатку)."""
    c = pcfg["control"]
    s = pcfg["stats"]
    with open(out / f"control_{rtag}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    lines = [f"# Контроль: заплатки только на фоне ({rtag})", "",
             f"Для каждой цели test — заплатка размера её маски уровня, поставленная вне всех рамок. Доля заплаток, "
             f"на которых есть детекция при замороженном пороге. Правило: treatment − control > {c['max_hit_gap']} "
             "-> полезность не устанавливается.", "",
             "| датасет | режим | seed | заплаток | с детекцией | доля | FPPI | Recall (цели открыты) |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['dataset']} | {r['regime']} | {r['seed']} | {r['patches']} | {r['patches_hit']} | "
              f"{r['hit_rate']:.3f} | {r['fppi']:.3f} | {r['recall']:.3f} |" for r in rows]
    lines += ["", "## Решение", ""]
    for ds, d in decision.items():
        mean = lambda reg: np.mean([r["hit_rate"] for r in rows if r["dataset"] == ds and r["regime"] == reg] or [0])
        gap = mean(s["treatment"]) - mean(s["control"])
        d["control_hit_gap"] = float(gap)
        d["shortcut_flag"] = bool(gap > c["max_hit_gap"])
        if d["shortcut_flag"]:
            d["useful"] = False
        lines.append(f"- **{ds}**: разность долей {s['treatment']} − {s['control']} = {gap:+.3f}; "
                     + ("**возможное обучение на заплатку — вывод о полезности не делается**."
                        if d["shortcut_flag"] else "в пределах допуска."))
    Path("reports").joinpath(f"control_{rtag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return decision


if __name__ == "__main__":
    main()
