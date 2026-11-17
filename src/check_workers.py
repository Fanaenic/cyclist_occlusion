"""Проверка воспроизводимости масок обучения при workers > 0 (fix_plan, шаг 6).

Два коротких обучения proposed с одним seed и одинаковым числом загрузчиков. Каждый процесс-загрузчик пишет
журнал: (эпоха, изображение, маски, приманки, хеш пикселей после наложения). Журналы двух запусков сравниваются
как мультимножества строк. Третий запуск с workers: 0 — для сравнения скорости.

Выход: reports/workers_check.md. Запуск: python src/check_workers.py --data runs/smoke/data/kitti.yaml
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def run_once(data, mcfg, tcfg, workers, epochs, seed, out, name, close_mosaic=0):
    from ultralytics import YOLO
    from masks import target_span_pool
    from occlusion_train import OcclusionTrainer

    log = out / name / "masklog"
    shutil.rmtree(out / name, ignore_errors=True)
    log.mkdir(parents=True)
    OcclusionTrainer.occ = {"regime": "proposed", "cfg": mcfg, "seed": mcfg["training"]["seed"] * 1000 + seed,
                            "span_pool": target_span_pool(mcfg, "kitti"), "mask_log": str(log.resolve())}
    t0 = time.time()
    YOLO(tcfg["model"]).train(trainer=OcclusionTrainer, data=data, epochs=epochs, imgsz=tcfg["imgsz"],
                              batch=tcfg["batch"], seed=seed, deterministic=True, workers=workers, amp=tcfg["amp"],
                              plots=False, val=False, close_mosaic=close_mosaic, project=str((out / name).resolve()), name="train",
                              exist_ok=True, verbose=False)
    dt = time.time() - t0
    lines = Counter()
    for f in log.glob("*.txt"):
        lines.update(f.read_text(encoding="utf-8").splitlines())
    return lines, dt


def digest(lines):
    return hashlib.sha256("\n".join(sorted(lines.elements())).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="runs/smoke/data/kitti.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timing-data", default="datasets_yolo/kitti.yaml",
                    help="полный набор для замера скорости одной эпохи (workers 0 и --workers)")
    args = ap.parse_args()
    os.environ["YOLO_OFFLINE"] = "1"
    with open("configs/masks.yaml", encoding="utf-8") as fh:
        mcfg = yaml.safe_load(fh)
    with open("configs/train.yaml", encoding="utf-8") as fh:
        tcfg = yaml.safe_load(fh)
    out = Path("runs/check_workers")
    a, ta = run_once(args.data, mcfg, tcfg, args.workers, args.epochs, args.seed, out, "a")
    b, tb = run_once(args.data, mcfg, tcfg, args.workers, args.epochs, args.seed, out, "b")
    z, tz = run_once(args.data, mcfg, tcfg, 0, args.epochs, args.seed, out, "w0")
    # скорость: одна эпоха на полном train
    _, t_full0 = run_once(args.timing_data, mcfg, tcfg, 0, 1, args.seed, out, "full_w0")
    _, t_fullw = run_once(args.timing_data, mcfg, tcfg, args.workers, 1, args.seed, out, "full_w")
    same = a == b
    per_epoch = Counter(ln.split("\t")[0] for ln in a.elements())
    lines = ["# Воспроизводимость масок обучения при workers > 0", "",
             f"Данные: `{args.data}`; режим proposed; seed {args.seed}; эпох {args.epochs}; workers {args.workers}. "
             "Журнал: эпоха, изображение, маски, приманки, хеш пикселей кадра после наложения масок "
             "(до mosaic и прочих аугментаций Ultralytics).", "",
             "| запуск | workers | записей | SHA-256 журнала | время, с |", "|---|---|---|---|---|",
             f"| A | {args.workers} | {sum(a.values())} | `{digest(a)[:16]}` | {ta:.0f} |",
             f"| B | {args.workers} | {sum(b.values())} | `{digest(b)[:16]}` | {tb:.0f} |",
             f"| для сравнения скорости | 0 | {sum(z.values())} | `{digest(z)[:16]}` | {tz:.0f} |", "",
             f"Записей по эпохам (A): {dict(sorted(per_epoch.items()))}. Mosaic включён с первой эпохи "
             "(close_mosaic: 0), поэтому в журнал попадают и плитки mosaic.", "",
             f"Скорость, одна эпоха на `{args.timing_data}`: workers 0 — {t_full0:.0f} с, "
             f"workers {args.workers} — {t_fullw:.0f} с (включая запуск процессов).", "",
             f"**Итог:** журналы A и B {'совпадают' if same else 'НЕ совпадают'}"
             + (f" -> workers: {args.workers} допустим." if same else " -> остаётся workers: 0.")]
    if not same:
        diff = (a - b) + (b - a)
        lines += ["", f"Различающихся записей: {sum(diff.values())}. Примеры:", "```"] + list(diff)[:5] + ["```"]
    Path("reports/workers_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
