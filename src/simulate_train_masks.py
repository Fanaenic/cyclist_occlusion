"""«Сухой» прогон обучающего генератора масок (fix_plan2, этап 1): одна эпоха по train без обучения.

Для каждого датасета и каждого режима с окклюзионной аугментацией (p > 0) повторяет то, что делает
src/occlusion_train.py на эпохе 0 для seed 1: тот же масштаб кадра (длинная сторона = imgsz), те же целевые рамки
и запрещённые области (datasets_yolo/<ds>/forbidden/train), тот же генератор (masks.plan_training_masks) и тот же
порядок изображений. Изображения не читаются — считается только план: сколько масок и приманок будет наложено,
сколько пропущено и по какой причине. Mosaic не моделируется (в обучении маски накладываются до него).

Выход: reports/train_masks_dryrun.md, reports/train_masks_dryrun.csv
Запуск: python src/simulate_train_masks.py [--dataset kitti|bdd] [--seed 1]
"""
import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from masks import plan_training_masks, target_span_pool  # noqa: E402
from occlusion_train import read_forbidden, to_pixel_box  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def image_sizes(ds, im_files, ycfg):
    if ds == "bdd":
        return {f: tuple(ycfg["bdd"]["image_size"]) for f in im_files}
    from PIL import Image
    out = {}
    for f in im_files:
        with Image.open(f) as im:
            out[f] = im.size
    return out


def simulate(ds, regime, seed, mcfg, tcfg, ycfg):
    root = Path(ycfg["out_dir"]) / ds
    im_files = sorted(str(p) for p in (root / "images" / "train").iterdir())     # порядок, как в Ultralytics
    sizes = image_sizes(ds, im_files, ycfg)
    span_pool = target_span_pool(mcfg, ds)
    imgsz = tcfg["imgsz"]
    st = Counter()
    for index, f in enumerate(im_files):
        lines = [ln.split() for ln in (root / "labels" / "train" / f"{Path(f).stem}.txt").read_text().splitlines()
                 if ln.strip()]
        st["images"] += 1
        if not lines:
            continue
        W0, H0 = sizes[f]
        r = imgsz / max(W0, H0)
        w, h = min(math.ceil(W0 * r), imgsz), min(math.ceil(H0 * r), imgsz)       # как load_image в Ultralytics
        boxes = []
        for _, cx, cy, bw, bh in (map(float, ln) for ln in lines):
            boxes.append(to_pixel_box(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h),
                                      w, h))
        sx, sy = w / W0, h / H0
        fb = [to_pixel_box((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), w, h) for b in read_forbidden(f)]
        rng = np.random.default_rng([mcfg["training"]["seed"] * 1000 + seed, index, 0])
        ops, skips = plan_training_masks(boxes, fb, w, h, rng, regime, mcfg, span_pool)
        st["images_with_targets"] += 1
        st["targets"] += len(boxes)
        st["forbidden_boxes"] += len(fb)
        st["masks"] += sum(op["kind"] == "mask" for op in ops)
        st["decoys"] += sum(op["kind"] == "decoy" for op in ops)
        for s in skips:
            st[f"skip:{s['kind']}:{s['reason']}"] += 1
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kitti", "bdd"], action="append")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    mcfg, tcfg, ycfg = load("configs/masks.yaml"), load("configs/train.yaml"), load("configs/yolo.yaml")
    regimes = [r for r in tcfg["regimes"] if mcfg["training"][r].get("p", 0) > 0]
    rows = []
    for ds in args.dataset or tcfg["datasets"]:
        for regime in regimes:
            st = simulate(ds, regime, args.seed, mcfg, tcfg, ycfg)
            rows.append({"dataset": ds, "regime": regime, **st})
            print(ds, regime, dict(st))
    skip_keys = sorted({k for r in rows for k in r if k.startswith("skip:")})
    fields = ["dataset", "regime", "images", "images_with_targets", "targets", "forbidden_boxes", "masks", "decoys"]
    rep = Path("reports")
    with open(rep / "train_masks_dryrun.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + skip_keys, restval=0)
        w.writeheader()
        w.writerows(rows)
    names = {"skip:mask:no_clean_source": "маска: нет чистого источника фона",
             "skip:mask:neighbor_cover": "маска: не обойти соседа",
             "skip:decoy:no_clean_place": "приманка: нет чистого места",
             "skip:decoy:no_clean_source": "приманка: нет чистого источника фона"}
    lines = ["# Сухой прогон обучающего генератора масок (одна эпоха train, без обучения)", "",
             f"Эпоха 0, seed {args.seed}, масштаб кадра — длинная сторона {tcfg['imgsz']} (как в обучении). "
             "Запрещённые области — все размеченные люди, rider, bike, cyclist и ignore-области кадра "
             "(datasets_yolo/<ds>/forbidden). Если чистого места нет, воздействие не применяется. "
             "Mosaic не моделируется: в обучении маски накладываются до него, на каждую плитку отдельно.", "",
             "| датасет | режим | кадров | с целями | целей | запрещённых рамок | масок | приманок | "
             + " | ".join(names.get(k, k) for k in skip_keys) + " |",
             "|---|---|---|---|---|---|---|---|" + "---|" * len(skip_keys)]
    lines += [f"| {r['dataset']} | {r['regime']} | {r['images']} | {r['images_with_targets']} | {r['targets']} | "
              f"{r['forbidden_boxes']} | {r['masks']} | {r['decoys']} | "
              + " | ".join(str(r.get(k, 0)) for k in skip_keys) + " |" for r in rows]
    lines += ["", "Ожидаемое число масок при отсутствии пропусков — около p · число целей (p = 0,5). У режимов с "
              "приманками число приманок не больше числа масок: приманка ставится только к наложенной маске и может "
              "быть пропущена отдельно."]
    (rep / "train_masks_dryrun.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"-> {rep / 'train_masks_dryrun.md'}")


if __name__ == "__main__":
    main()
