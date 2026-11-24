"""Весь конвейер: обучение -> порог на calib -> оценка на всех уровнях масок -> бутстрэп.

Для каждого датасета, режима (configs/train.yaml, regimes) и seed:
  1. обучение YOLOv8n (src/occlusion_train.py — маски режима при обучении; пропуски — в skip_log);
  2. детекции на calib без масок -> порог для FPPI = target_fppi (заморожен);
  3. детекции на оцениваемой части для каждого уровня маски (маски из splits/masks_*_<часть>.csv, одни и те же
     для всех моделей) -> src/evaluate.py: метрики и .npz по изображениям;
  4. src/stats.py по всем .npz.

Оцениваемая часть: основная серия — test; технический прогон (--smoke) — ТОЛЬКО calib (замечание 5):
изображения test в технический прогон не входят. В техническом прогоне порог и оценка — на одной и той же calib,
это проверка конвейера, а не результат.

Каждый запуск сохраняет снимок {out_dir}/{tag}/snapshots/<время>/: все configs/*.yaml, git hash и список
изменённых файлов, версии пакетов, командную строку и списки использованных изображений (замечание 2).

Выход: {out_dir}/{tag}/ — веса, детекции, eval/*.npz, eval/thresholds.json, metrics.csv; reports/stats_{tag}.*

Запуск:
  python src/run_experiment.py --smoke                     технический прогон (limit и epochs из smoke), только calib
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate, ground_truth, load_gt  # noqa: E402
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


def build_subset(ycfg, ds, limit, out, parts):
    """Набор для Ultralytics: при limit — подмножество частей parts (жёсткие ссылки), иначе — datasets_yolo как есть."""
    root = Path(ycfg["out_dir"]) / ds
    images = {p: subset_images(root, p, limit) for p in parts}
    if not limit:
        return Path(ycfg["out_dir"]) / f"{ds}.yaml", images, root
    sub = out / "data" / ds
    for p in parts:
        for d in ("images", "labels", "forbidden"):
            (sub / d / p).mkdir(parents=True, exist_ok=True)
        files = {f.stem: f for f in (root / "images" / p).iterdir()}
        for stem in images[p]:
            src = files[stem]
            dst = sub / "images" / p / src.name
            if not dst.exists():
                os.link(src, dst)
            for d in ("labels", "forbidden"):
                (sub / d / p / f"{stem}.txt").write_bytes((root / d / p / f"{stem}.txt").read_bytes())
    data_yaml = out / "data" / f"{ds}.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(sub.resolve()), "train": "images/train", "val": "images/calib",
                                         "names": {0: ycfg["class_name"]}}), encoding="utf-8")
    return data_yaml, images, root


def save_snapshot(out, images_by_ds):
    """Снимок запуска: конфиги, git hash и изменённые файлы, версии, командная строка, списки изображений."""
    import platform
    import shutil
    import subprocess

    snap = out / "snapshots" / time.strftime("%Y%m%d-%H%M%S")
    (snap / "configs").mkdir(parents=True, exist_ok=True)
    for f in sorted(Path("configs").glob("*.yaml")):
        shutil.copy2(f, snap / "configs" / f.name)

    def git(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as e:
            return f"недоступно: {e}"

    (snap / "git.txt").write_text(f"commit: {git('rev-parse', 'HEAD')}\nизменённые файлы (git status --porcelain):\n"
                                  f"{git('status', '--porcelain') or '(нет)'}\n", encoding="utf-8")
    versions = {"python": platform.python_version()}
    for mod in ("torch", "ultralytics", "numpy", "cv2"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = "нет"
    (snap / "run.json").write_text(json.dumps({"argv": sys.argv, "versions": versions}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    for ds, parts in images_by_ds.items():
        for part, names in parts.items():
            (snap / f"images_{ds}_{part}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    return snap


# ---------------------------------------------------------------- обучение

def train(tcfg, mcfg, data_yaml, ds, regime, seed, epochs, out):
    from ultralytics import YOLO
    from occlusion_train import OcclusionTrainer

    name = f"{ds}__{regime}__s{seed}"
    weights = out / "train" / name / "weights" / "last.pt"      # модель — веса последней эпохи (см. train.yaml)
    done = out / "train" / name / "done.json"                   # метка завершённого обучения
    if done.exists():
        return weights
    from masks import target_span_pool
    skip_log = out / "train" / name / "skip_log"
    if skip_log.exists():                                       # незавершённое обучение начинается заново
        for f in skip_log.glob("*.txt"):
            f.unlink()
    skip_log.mkdir(parents=True, exist_ok=True)
    OcclusionTrainer.occ = {"regime": regime, "cfg": mcfg, "seed": mcfg["training"]["seed"] * 1000 + seed,
                            "span_pool": target_span_pool(mcfg, ds), "skip_log": str(skip_log.resolve())}
    model = YOLO(tcfg["model"])
    model.train(trainer=OcclusionTrainer, data=str(data_yaml), epochs=epochs, imgsz=tcfg["imgsz"],
                batch=tcfg["batch"], seed=seed, deterministic=tcfg["deterministic"], workers=tcfg["workers"],
                amp=tcfg["amp"], plots=tcfg["plots"], val=tcfg["val"], project=str((out / "train").resolve()),
                name=name, exist_ok=True, verbose=False)
    skips = {}
    for f in skip_log.glob("*.txt"):
        for ln in f.read_text(encoding="utf-8").splitlines():
            _, _, kind, reason = ln.split("\t")
            skips[f"{kind}:{reason}"] = skips.get(f"{kind}:{reason}", 0) + 1
    done.write_text(json.dumps({"epochs": epochs, "seed": seed, "regime": regime, "skipped": skips},
                               ensure_ascii=False), encoding="utf-8")
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


def read_masks(splits_dir, ds, level, part):
    """{stem: [((x0,y0,x1,y1), mask_seed, (sx, sy))]} для уровня (уровень 0 — пусто)."""
    res = {}
    if level == 0:
        return res
    with open(Path(splits_dir) / f"masks_{ds}_{part}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if float(r["level"]) == level:
                res.setdefault(Path(r["image"]).stem, []).append(
                    (tuple(int(r[k]) for k in ("mx0", "my0", "mx1", "my1")), int(r["mask_seed"]),
                     (r["sx"], r["sy"])))
    return res


def read_control(splits_dir, ds, part):
    """Контрольные заплатки на фоне: {stem: [((x0,y0,x1,y1), mask_seed, (sx, sy))]}."""
    res = {}
    with open(Path(splits_dir) / f"control_{ds}_{part}.csv", encoding="utf-8") as fh:
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


def evaluate_part(dets, ds, part, images, level, subset, pcfg, ycfg, population=None, loaded=None):
    """Оценка детекций части на уровне маски; population — совокупность целей (по умолчанию из протокола)."""
    gt = ground_truth(ds, part, level, images, pcfg, ycfg, subset, population, loaded)
    ev, _ = evaluate(dets, gt, images, pcfg)
    ev.target_boxes = {img: [list(map(float, box)) for box in sorted(gt[img][0])] for img in images}
    return ev


# Оцениваемые варианты test: (подмножество, совокупность, суффикс имени .npz). Первый — основной.
VARIANTS = [("all", "all_levels", "all"), ("not_occluded", "all_levels", "not_occluded"),
            ("all", "per_level", "all-per_level")]


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
    thr_path = ev_dir / "thresholds.json"
    thresholds = json.loads(thr_path.read_text()) if thr_path.exists() else {}
    metrics_rows, control_rows, timing = [], [], {}
    # технический прогон оценивает только calib; test в нём не участвует (замечание 5)
    part = "calib" if args.smoke else "test"
    parts = ["train", "calib"] if args.smoke else PARTS
    subsets = {ds: build_subset(ycfg, ds, limit, out, parts) for ds in datasets}
    snap = save_snapshot(out, {ds: s[1] for ds, s in subsets.items()})
    print(f"Снимок конфигураций и списков изображений: {snap}; оцениваемая часть: {part}")

    for ds in datasets:
        data_yaml, images, root = subsets[ds]
        img_files = {p: {f.stem: f for f in (root / "images" / p).iterdir()} for p in ("calib", part)}
        print(f"[{ds}] " + ", ".join(f"{p} = {len(images[p])}" for p in parts) + " изображений")
        for regime in regimes:
            for seed in tcfg["seeds"]:
                key = f"{ds}__{regime}__s{seed}"
                t0 = time.time()
                w = train(tcfg, mcfg, data_yaml, ds, regime, seed, epochs, out)
                timing[f"{key} train"] = time.time() - t0
                t0 = time.time()
                calib_paths = [img_files["calib"][s] for s in images["calib"]]
                # порог — на calib без масок; масок нет, поэтому совокупность целей — все цели (per_level)
                ev = evaluate_part(detect(w, calib_paths, {}, tcfg, pcfg, mcfg["fill"]), ds, "calib",
                                   images["calib"], 0.0, "all", pcfg, ycfg, population="per_level")
                thresholds[key] = ev.threshold_for_fppi(pcfg["metrics"]["target_fppi"])
                thr_path.write_text(json.dumps(thresholds, indent=1))
                test_paths = [img_files[part][s] for s in images[part]]
                test_gt = load_gt(ds, part, ycfg)
                dets0 = None
                # основные маски (заливка fill) + те же прямоугольники с шумовой заливкой (абляция и схема 2×2)
                for level, fill_name in [(L, "bg") for L in levels] + [(L, "noise") for L in levels]:
                    fill = mcfg["fill"] if fill_name == "bg" else mcfg["test_noise_fill"]
                    if level == 0 and fill_name == "noise":
                        dets = dets0                         # без масок заливка не важна
                    else:
                        dets = detect(w, test_paths, read_masks(mcfg["splits_dir"], ds, level, part), tcfg, pcfg,
                                      fill)
                    if level == 0:
                        dets0 = dets
                    for subset, population, name in VARIANTS:
                        if subset == "not_occluded" and ds != "bdd":
                            continue
                        if fill_name == "noise":
                            if name != "all":
                                continue
                            name = "all-noise"
                        ev = evaluate_part(dets, ds, part, images[part], level, subset, pcfg, ycfg,
                                           population, test_gt)
                        ev.save(ev_dir / f"{key}__L{level}__{name}.npz", dataset=ds, split=part, regime=regime,
                                seed=seed, level=level, subset=subset, population=population,
                                fill=fill_name, threshold=thresholds[key], target_boxes=ev.target_boxes)
                        at = ev.at_threshold(thresholds[key])
                        metrics_rows.append({"dataset": ds, "regime": regime, "seed": seed, "level": level,
                                             "subset": name, "threshold": thresholds[key], "targets": ev.n_targets,
                                             "images": ev.n_images, **at, "ap50": ev.ap(),
                                             "recall_at_fppi_curve": ev.recall_at_fppi(pcfg["metrics"]["target_fppi"])})
                        print(f"  {key} L={level:.0%} {name:14s} R={at['recall']:.3f} FPPI={at['fppi']:.3f} "
                              f"AP={ev.ap():.3f} thr={thresholds[key]:.3f}")
                # контроль: заплатки только на фоне, цели открыты; заливка — фоном (основной) и шумом (абляция)
                ctrl = {s: v for s, v in read_control(mcfg["splits_dir"], ds, part).items() if s in set(images[part])}
                cpaths = [img_files[part][s] for s in sorted(ctrl)]
                for fill_name, fill in (("bg", mcfg["fill"]), ("noise", mcfg["test_noise_fill"])):
                    cdets = detect(w, cpaths, ctrl, tcfg, pcfg, fill)
                    cres = control_check(cdets, ctrl, thresholds[key], pcfg)
                    ev0 = evaluate_part(cdets, ds, part, sorted(ctrl), 0.0, "all", pcfg, ycfg, loaded=test_gt)
                    cres.update(dataset=ds, split=part, regime=regime, seed=seed, fill=fill_name,
                                fppi=ev0.at_threshold(thresholds[key])["fppi"],
                                recall=ev0.at_threshold(thresholds[key])["recall"])
                    control_rows.append(cres)
                    print(f"  {key} control ({fill_name}): заплаток {cres['patches']}, с детекцией "
                          f"{cres['patches_hit']} ({cres['hit_rate']:.3f}); FPPI={cres['fppi']:.3f}")
                timing[f"{key} eval"] = time.time() - t0

    with open(out / f"metrics_{rtag}.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(metrics_rows[0].keys()))
        wr.writeheader()
        wr.writerows(metrics_rows)
    (out / f"timing_{rtag}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")

    from stats import analyse, factorial_report
    from finalize import finalize, write_final
    override = {"datasets": datasets, "split": part}
    if args.replicas:
        override["bootstrap"] = {**pcfg["stats"]["bootstrap"], "replicas": args.replicas}
    t0 = time.time()
    _, decision = analyse(ev_dir, pcfg, override, tag=rtag)
    if "ore" in regimes:
        analyse(ev_dir, pcfg, override, tag=f"{rtag}_vs_ore", control="ore")
    if any(ds == "bdd" for ds in datasets):
        analyse(ev_dir, pcfg, {**override, "datasets": ["bdd"], "subset": "not_occluded"},
                tag=f"{rtag}_not_occluded")
    # дополнительный отчёт: цели исключаются только на тех уровнях, где их закрывают маски соседей
    analyse(ev_dir, pcfg, {**override, "subset": "all-per_level"}, tag=f"{rtag}_per_level")
    if "proposed_noise" in regimes:
        # абляция утечки признака: proposed_noise против baseline на фоновых и на шумовых тестовых масках
        analyse(ev_dir, pcfg, override, tag=f"{rtag}_noise_ablation_bg", treatment="proposed_noise")
        analyse(ev_dir, pcfg, {**override, "subset": "all-noise"}, tag=f"{rtag}_noise_ablation_noise",
                treatment="proposed_noise")
    if set(pcfg["factorial"]["regimes"].values()) <= set(regimes):
        factorial_report(ev_dir, out, pcfg, {**pcfg["stats"], **override}, datasets, part, control_rows)
    timing["stats"] = time.time() - t0
    (out / f"timing_{rtag}.json").write_text(json.dumps(timing, indent=1), encoding="utf-8")
    with (out / f"control_{rtag}.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(control_rows[0]))
        wr.writeheader()
        wr.writerows(control_rows)
    results = finalize(decision, control_rows, pcfg, pcfg["stats"], part)
    write_final(results, out)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
