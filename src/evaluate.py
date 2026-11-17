"""Оценщик: Recall при замороженном пороге, фактический FPPI, Recall@FPPI по кривой, AP@0.5.

Все правила — в configs/protocol.yaml (NMS, порядок сопоставления, ignore, дубли, правило разного FPPI).

Библиотека:
  nms(dets, cfg)                                  NMS протокола (одинаковый для всех моделей)
  match_image(dets, targets, ignores, cfg)        TP / FP / ignore для детекций одного изображения
  Evaluation(...)                                 кривая Recall–FPPI, порог для FPPI, AP
  level_ground_truth(...)                         цели и ignore на заданном уровне маски и подмножестве
  compare_fppi_rule(a, b, thr_a, thr_b, cfg)      правило при различии фактического FPPI

Командная строка (предсказания — CSV: image,x1,y1,x2,y2,score; image — имя файла без расширения):
  python src/evaluate.py --dataset kitti --part calib --level 0 --pred preds.csv --calibrate
  python src/evaluate.py --dataset kitti --part test --level 0.5 --pred preds.csv --threshold 0.31
  python src/evaluate.py --dataset bdd --part test --level 0 --gt-as-pred     # проверка конвейера
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------- геометрия

def iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ioa(det, region):
    """Доля площади детекции внутри области."""
    iw = min(det[2], region[2]) - max(det[0], region[0])
    ih = min(det[3], region[3]) - max(det[1], region[1])
    a = (det[2] - det[0]) * (det[3] - det[1])
    return iw * ih / a if iw > 0 and ih > 0 and a > 0 else 0.0


# ---------------------------------------------------------------- NMS и сопоставление

def nms(dets, cfg):
    """dets: [(x1, y1, x2, y2, score)]. Порог conf -> жадный NMS по убыванию score -> max_det."""
    c = cfg["nms"]
    order = sorted((d for d in dets if d[4] >= c["conf"]), key=lambda d: -d[4])
    keep = []
    for d in order:
        if all(iou(d, k) <= c["iou"] for k in keep):
            keep.append(d)
            if len(keep) == c["max_det"]:
                break
    return keep


def match_image(dets, targets, ignores, cfg):
    """Возвращает [(score, 'tp'|'fp'|'ign')] и множество найденных целей (индексы в targets)."""
    m = cfg["matching"]
    taken, out = set(), []
    for d in sorted(dets, key=lambda d: -d[4]):          # sorted устойчив: равные score — в исходном порядке
        best, best_iou, hits_taken = None, -1.0, False
        for j, t in enumerate(targets):
            v = iou(d, t)
            if v < m["iou"]:
                continue
            if j in taken:
                hits_taken = True
            elif v > best_iou:                           # при равном IoU остаётся цель с меньшим индексом
                best, best_iou = j, v
        if best is not None:
            taken.add(best)
            out.append((d[4], "tp"))
        elif hits_taken and m["duplicates"] == "fp":
            out.append((d[4], "fp"))
        elif any(ioa(d, g) >= m["ignore_ioa"] for g in ignores):
            out.append((d[4], "ign"))
        else:
            out.append((d[4], "fp"))
    return out, taken


# ---------------------------------------------------------------- метрики

class Evaluation:
    """Накопленные результаты по всем изображениям части."""

    def __init__(self, n_images, n_targets):
        self.n_images, self.n_targets = n_images, n_targets
        self.scores, self.labels, self.det_images = [], [], []
        self.targets_per_image = {}

    def add(self, matched, image=None):
        for s, lab in matched:
            if lab != "ign":
                self.scores.append(s)
                self.labels.append(lab == "tp")
                self.det_images.append(image)

    def save(self, path, **meta):
        """Результат по изображениям для src/stats.py: детекции (после NMS и сопоставления) и число целей."""
        images = sorted(self.targets_per_image)
        idx = {img: i for i, img in enumerate(images)}
        np.savez_compressed(
            path, images=np.array(images), n_targets=np.array([self.targets_per_image[i] for i in images]),
            det_image=np.array([idx[i] for i in self.det_images], dtype=np.int32),
            score=np.asarray(self.scores, dtype=np.float32), tp=np.asarray(self.labels, dtype=bool),
            meta=json.dumps(meta, ensure_ascii=False))

    def _sorted(self):
        s = np.asarray(self.scores, dtype=float)
        tp = np.asarray(self.labels, dtype=bool)
        order = np.argsort(-s, kind="stable")
        return s[order], tp[order]

    def curve(self):
        """Точки (порог, recall, fppi) для каждого уникального порога: детекции со score >= порога."""
        s, tp = self._sorted()
        if len(s) == 0:
            return np.array([]), np.array([]), np.array([])
        ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
        last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # последний индекс каждой группы равных score
        return s[last], ctp[last] / max(self.n_targets, 1), cfp[last] / self.n_images

    def at_threshold(self, thr):
        s, tp = self._sorted()
        keep = s >= thr
        return {"recall": float(tp[keep].sum() / max(self.n_targets, 1)),
                "fppi": float((~tp[keep]).sum() / self.n_images),
                "tp": int(tp[keep].sum()), "fp": int((~tp[keep]).sum())}

    def threshold_for_fppi(self, target):
        """Наименьший порог, при котором FPPI <= target. Если даже первый порог даёт больше — +inf."""
        thr, _, fppi = self.curve()
        ok = np.nonzero(fppi <= target + 1e-12)[0]
        return float(thr[ok[-1]]) if len(ok) else float("inf")

    def recall_at_fppi(self, target):
        """Наибольший Recall на кривой при FPPI <= target (без заморозки порога — диагностика)."""
        _, rec, fppi = self.curve()
        ok = fppi <= target + 1e-12
        return float(rec[ok].max()) if ok.any() else 0.0

    def ap(self):
        """AP@IoU протокола: площадь под огибающей precision–recall (all-point interpolation)."""
        s, tp = self._sorted()
        if self.n_targets == 0:
            return float("nan")
        if len(s) == 0:
            return 0.0
        ctp, cfp = np.cumsum(tp), np.cumsum(~tp)
        rec = ctp / self.n_targets
        prec = ctp / (ctp + cfp)
        mrec = np.r_[0.0, rec, 1.0]
        mpre = np.r_[1.0, prec, 0.0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(dets_by_image, gt_by_image, images, cfg):
    """dets_by_image: {img: [(x1,y1,x2,y2,score)]}; gt_by_image: {img: (targets, ignores)}."""
    n_t = sum(len(gt_by_image.get(i, ([], []))[0]) for i in images)
    ev = Evaluation(len(images), n_t)
    found = {}
    for img in images:
        targets, ignores = gt_by_image.get(img, ([], []))
        matched, taken = match_image(nms(dets_by_image.get(img, []), cfg), targets, ignores, cfg)
        ev.add(matched, img)
        ev.targets_per_image[img] = len(targets)
        found[img] = taken
    return ev, found


def compare_fppi_rule(ev_a, ev_b, thr_a, thr_b, cfg):
    """Правило протокола для пары моделей на одном test (см. protocol.yaml: fppi_rule)."""
    r = cfg["fppi_rule"]
    a, b = ev_a.at_threshold(thr_a), ev_b.at_threshold(thr_b)
    diff = a["recall"] - b["recall"]
    res = {"recall_a": a["recall"], "recall_b": b["recall"], "fppi_a": a["fppi"], "fppi_b": b["fppi"],
           "diff": diff, "fppi_gap": abs(a["fppi"] - b["fppi"]), "budget_matched": False}
    if res["fppi_gap"] <= r["fppi_tolerance"]:
        res["verdict"] = "a" if diff > 0 else "b" if diff < 0 else "tie"
        return res
    budget = max(a["fppi"], b["fppi"]) if r["matched_budget"] == "max" else min(a["fppi"], b["fppi"])
    ra, rb = ev_a.recall_at_fppi(budget), ev_b.recall_at_fppi(budget)
    res.update(budget_matched=True, budget=budget, recall_a_matched=ra, recall_b_matched=rb)
    same_sign = np.sign(diff) == np.sign(ra - rb) and diff != 0
    res["verdict"] = ("a" if diff > 0 else "b") if same_sign else "not_established"
    return res


# ---------------------------------------------------------------- цели на уровне маски

def covered_fraction(box, masks):
    """Точная доля пиксельной рамки, закрытая объединением масок (маски в целых пикселях)."""
    x0, y0 = int(np.floor(box[0])), int(np.floor(box[1]))
    x1, y1 = int(np.ceil(box[2])), int(np.ceil(box[3]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    grid = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for m in masks:
        ax0, ay0 = max(m[0], x0) - x0, max(m[1], y0) - y0
        ax1, ay1 = min(m[2], x1) - x0, min(m[3], y1) - y0
        if ax1 > ax0 and ay1 > ay0:
            grid[ay0:ay1, ax0:ax1] = True
    return float(grid.mean())


def neighbor_covered(t, masks, cfg):
    """Закрывают ли маски СОСЕДЕЙ цель t сильнее neighbor_mask_max_cover (masks — {object_id: маска} уровня)."""
    others = [m for oid, m in masks.items() if oid != t["id"]]
    return bool(others) and covered_fraction(t["box"], others) > cfg["matching"]["neighbor_mask_max_cover"]


def valid_on_all_levels(targets, masks_by_level, cfg):
    """id целей, которые ни на одном уровне не закрыты масками соседей сверх порога (основная совокупность)."""
    bad = set()
    for by_img in masks_by_level.values():
        for img, ms in by_img.items():
            bad |= {t["id"] for t in targets.get(img, []) if neighbor_covered(t, ms, cfg)}
    return {t["id"] for ts in targets.values() for t in ts} - bad


def level_ground_truth(targets, ignores, masks, cfg, subset=None, valid_ids=None):
    """targets: [{'id', 'box', 'occluded'}]; masks: {object_id: mask_box} для этого уровня.

    Возвращает (рамки целей, рамки ignore, id целей). Цель уходит в ignore, если она вне подмножества,
    вне valid_ids (основная совокупность — цели, валидные на всех уровнях) или маски СОСЕДЕЙ закрывают её
    на этом уровне сильнее neighbor_mask_max_cover.
    """
    keep, ign, ids = [], list(ignores), []
    for t in targets:
        in_subset = (subset is None or subset(t)) and (valid_ids is None or t["id"] in valid_ids)
        if not in_subset or neighbor_covered(t, masks, cfg):
            ign.append(t["box"])
        else:
            keep.append(t["box"])
            ids.append(t["id"])
    return keep, ign, ids


# ---------------------------------------------------------------- загрузка данных

SUBSETS = {
    "all": None,
    "not_occluded": lambda t: not t["occluded"],
}


def load_gt(ds, part, ycfg):
    """Изображения части (как в datasets_yolo), цели с id и признаком occluded, ignore, маски по уровням."""
    root = Path(ycfg["out_dir"]) / ds
    images = sorted(p.stem for p in (root / "labels" / part).glob("*.txt"))
    img_set = set(images)
    targets = defaultdict(list)
    if ds == "kitti":
        with open(ycfg["kitti"]["selected_file"], encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["image"] in img_set:
                    targets[r["image"]].append({
                        "id": f"{r['image']}:{r['line']}", "occluded": int(r["occluded"]) != 0,
                        "box": tuple(float(r[k]) for k in ("left", "top", "right", "bottom"))})
    else:
        b = ycfg["bdd"]
        with open(b["selected_file"].format(source=b["sources"][part]), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                stem = Path(r["image"]).stem
                if stem in img_set and r["last_step"] == b["final_step"]:
                    targets[stem].append({
                        "id": r["rider_id"], "occluded": r["rider_occluded"] == "True",
                        "box": tuple(float(r[f"target_{k}"]) for k in ("x1", "y1", "x2", "y2"))})
    ignores = {}
    for img in images:
        lines = (root / "ignore" / part / f"{img}.txt").read_text(encoding="utf-8").splitlines()
        ignores[img] = [tuple(map(float, ln.split()[:4])) for ln in lines if ln.strip()]
    masks = defaultdict(lambda: defaultdict(dict))
    mpath = Path(ycfg["splits_dir"]) / f"masks_{ds}_{part}.csv"
    if mpath.exists():
        with open(mpath, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if float(r["level"]) > 0:
                    masks[float(r["level"])][Path(r["image"]).stem][r["object_id"]] = \
                        tuple(int(r[k]) for k in ("mx0", "my0", "mx1", "my1"))
    return images, targets, ignores, masks


def read_predictions(path):
    dets = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            dets[Path(r["image"]).stem].append(tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2", "score")))
    return dets


def ground_truth(ds, part, level, images, cfg, ycfg, subset="all", population=None, loaded=None):
    """{img: (цели, ignore)} для части, уровня, подмножества и совокупности (matching.population).

    population: all_levels — только цели, валидные на всех уровнях (остальные — ignore на всех уровнях);
    per_level — цель исключается только на тех уровнях, где её закрывают маски соседей.
    """
    population = population or cfg["matching"]["population"]
    _, targets, ignores, masks = loaded or load_gt(ds, part, ycfg)
    valid = valid_on_all_levels(targets, masks, cfg) if population == "all_levels" else None
    gt = {}
    for img in images:
        keep, ign, _ = level_ground_truth(targets.get(img, []), ignores.get(img, []),
                                          masks[level][img] if level > 0 else {}, cfg, SUBSETS[subset], valid)
        gt[img] = (keep, ign)
    return gt


def population_report(cfg, ycfg, report_dir="reports"):
    """Число целей основной совокупности (валидны на всех уровнях) и исключённых по уровням."""
    rows = []
    for ds in ("kitti", "bdd"):
        for part in ("calib", "test"):
            images, targets, _, masks = load_gt(ds, part, ycfg)
            total = sum(len(v) for v in targets.values())
            valid = valid_on_all_levels(targets, masks, cfg)
            vis = sum(1 for ts in targets.values() for t in ts if not t["occluded"])
            vis_valid = sum(1 for ts in targets.values() for t in ts if not t["occluded"] and t["id"] in valid)
            per_level = {L: sum(neighbor_covered(t, masks[L].get(img, {}), cfg)
                                for img, ts in targets.items() for t in ts) for L in sorted(masks)}
            rows.append({"dataset": ds, "part": part, "targets": total, "valid_all_levels": len(valid),
                         "not_occluded": vis, "not_occluded_valid_all_levels": vis_valid,
                         **{f"excluded_at_{int(L * 100)}": n for L, n in per_level.items()}})
    rep = Path(report_dir)
    with open(rep / "population.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lv = [k for k in rows[0] if k.startswith("excluded_at_")]
    lines = ["# Совокупность целей для кривой деградации", "",
             f"Основная совокупность (`matching.population: {cfg['matching']['population']}`): цели, которые ни на одном "
             f"уровне не закрыты масками соседей сверх {cfg['matching']['neighbor_mask_max_cover']:.0%} площади; "
             "остальные — ignore на всех уровнях. Так кривая 0→90 % строится по одним и тем же объектам. "
             "Вариант «по уровню» (исключение только на своём уровне) — дополнительный отчёт.", "",
             "| датасет | часть | целей | основная совокупность | occluded=false | occluded=false в основной | "
             + " | ".join(f"исключено по уровню {k.split('_')[-1]} %" for k in lv) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(lv)]
    lines += [f"| {r['dataset']} | {r['part']} | {r['targets']} | {r['valid_all_levels']} | {r['not_occluded']} | "
              f"{r['not_occluded_valid_all_levels']} | " + " | ".join(str(r[k]) for k in lv) + " |" for r in rows]
    (rep / "population.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return rows


def run(ds, part, level, dets, cfg, ycfg, subset="all", population=None):
    images = load_gt(ds, part, ycfg)[0]
    gt = ground_truth(ds, part, level, images, cfg, ycfg, subset, population)
    ev, _ = evaluate(dets, gt, images, cfg)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kitti", "bdd"])
    ap.add_argument("--part", choices=["calib", "test"])
    ap.add_argument("--level", type=float, default=0.0)
    ap.add_argument("--subset", choices=list(SUBSETS), default="all")
    ap.add_argument("--population", choices=["all_levels", "per_level"], help="по умолчанию — из протокола")
    ap.add_argument("--population-report", action="store_true",
                    help="только посчитать основную совокупность целей -> reports/population.*")
    ap.add_argument("--pred", help="CSV image,x1,y1,x2,y2,score")
    ap.add_argument("--gt-as-pred", action="store_true", help="подставить цели как детекции (проверка конвейера)")
    ap.add_argument("--threshold", type=float, help="замороженный порог confidence")
    ap.add_argument("--calibrate", action="store_true", help="подобрать порог для target_fppi на этой части")
    ap.add_argument("--save", help="сохранить результат по изображениям (.npz) для src/stats.py")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--yolo-config", default="configs/yolo.yaml")
    args = ap.parse_args()
    with open(args.protocol, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with open(args.yolo_config, encoding="utf-8") as fh:
        ycfg = yaml.safe_load(fh)
    if args.population_report:
        population_report(cfg, ycfg)
        return
    if not (args.dataset and args.part):
        ap.error("нужны --dataset и --part")

    if args.gt_as_pred:
        _, targets, _, _ = load_gt(args.dataset, args.part, ycfg)
        dets = {img: [(*t["box"], 0.9) for t in ts] for img, ts in targets.items()}
    else:
        dets = read_predictions(args.pred)
    ev = run(args.dataset, args.part, args.level, dets, cfg, ycfg, args.subset, args.population)
    out = {"dataset": args.dataset, "part": args.part, "level": args.level, "subset": args.subset,
           "population": args.population or cfg["matching"]["population"],
           "images": ev.n_images, "targets": ev.n_targets, "ap50": ev.ap(),
           "recall_at_target_fppi_curve": ev.recall_at_fppi(cfg["metrics"]["target_fppi"])}
    if args.calibrate:
        out["threshold"] = ev.threshold_for_fppi(cfg["metrics"]["target_fppi"])
    thr = out.get("threshold", args.threshold)
    if thr is not None:
        out.update({f"{k}_at_threshold": v for k, v in ev.at_threshold(thr).items()}, threshold=thr)
    if args.save:
        ev.save(args.save, dataset=args.dataset, part=args.part, level=args.level, subset=args.subset)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
