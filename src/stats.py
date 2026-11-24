"""Статистическая процедура: 3 парных seed + кластерный бутстрэп + Бонферрони, единое правило решения.

Процедура и её обоснование — text/stats.md; параметры — configs/protocol.yaml (раздел stats).

Вход (каталог --runs):
  {dataset}__{regime}__s{seed}__L{level}__{subset}.npz   результат src/evaluate.py --save на test
  thresholds.json   {"{dataset}__{regime}__s{seed}": порог, замороженный на calib}

Выход:
  reports/stats_{tag}.csv   оценка и интервал для каждой проверки
  reports/stats_{tag}.md    таблица и итоговое решение по правилу

Запуск: python src/stats.py --runs runs/eval [--tag main] [--replicas N]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


# ---------------------------------------------------------------- данные одного прогона

class Run:
    """Детекции одной модели на одном уровне маски, отсортированные по убыванию score, с номерами кластеров."""

    def __init__(self, path, cluster_of, clusters, expected=None):
        with np.load(path) as archive:
            if expected and "meta" not in archive:
                raise ValueError(f"{path}: отсутствуют метаданные")
            z = {key: archive[key] for key in archive.files}
        meta = json.loads(z["meta"].item()) if expected else {}
        for key, value in (expected or {}).items():
            if meta.get(key) != value:
                raise ValueError(f"{path}: ожидалось {key}={value!r}")
        images = [str(i) for i in z["images"]]
        if len(set(images)) != len(images) or set(images) != set(cluster_of):
            raise ValueError(f"{path}: несовместимый список изображений")
        self.target_counts = dict(zip(images, z["n_targets"], strict=True))
        self.target_boxes = meta.get("target_boxes")
        if expected and (not isinstance(self.target_boxes, dict)
                         or self.target_counts != {img: len(boxes) for img, boxes in self.target_boxes.items()}):
            raise ValueError(f"{path}: отсутствует или не согласован список целевых рамок")
        cl = np.array([clusters[cluster_of[i]] for i in images], dtype=np.int64)
        order = np.argsort(-z["score"], kind="stable")
        self.score = z["score"][order].astype(float)
        self.tp = z["tp"][order]
        self.det_cluster = cl[z["det_image"][order]]
        C = len(clusters)
        self.targets_c = np.bincount(cl, weights=z["n_targets"], minlength=C)
        self.images_c = np.bincount(cl, minlength=C).astype(float)
        last = np.r_[np.nonzero(np.diff(self.score))[0], len(self.score) - 1] if len(self.score) else np.array([], int)
        self.group_end = last                      # последний индекс каждой группы равных score

    def metrics(self, W, thr, budget=None):
        """W — веса кластеров (k x C). Возвращает массивы длины k: recall, fppi при пороге thr, ap, recall@budget."""
        Nt = W @ self.targets_c
        Ni = W @ self.images_c
        k = W.shape[0]
        Nt = np.where(Nt > 0, Nt, np.nan)
        Ni = np.where(Ni > 0, Ni, np.nan)
        if len(self.score) == 0:
            return {"recall": 0 / Nt, "fppi": 0 / Ni, "ap": 0 / Nt, "recall_budget": 0 / Nt}
        w = W[:, self.det_cluster]
        ctp = np.cumsum(w * self.tp, axis=1)
        cfp = np.cumsum(w * ~self.tp, axis=1)
        n = int(np.searchsorted(-self.score, -thr, side="right"))   # детекций со score >= thr
        rec_thr = (ctp[:, n - 1] if n else 0) / Nt
        fppi_thr = (cfp[:, n - 1] if n else 0) / Ni
        rec = ctp / Nt[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.nan_to_num(ctp / (ctp + cfp))
        env = np.maximum.accumulate(prec[:, ::-1], axis=1)[:, ::-1]
        ap = np.sum(np.diff(np.c_[np.zeros(k), rec], axis=1) * env, axis=1)
        out = {"recall": rec_thr, "fppi": fppi_thr, "ap": ap}
        if budget is not None:
            ge = self.group_end
            ok = (cfp[:, ge] / Ni[:, None]) <= budget + 1e-12
            cnt = ok.sum(axis=1)
            vals = (ctp[:, ge] / Nt[:, None])
            out["recall_budget"] = np.where(cnt > 0, vals[np.arange(k), np.maximum(cnt - 1, 0)], 0 / Nt)
        return out


def load_clusters(path, images=None, part="test"):
    """Кластеры части part (test; в техническом прогоне — calib); при images — только оцениваемых изображений."""
    cluster_of = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            stem = Path(r["image"]).stem
            if r["part"] == part and (images is None or stem in images):
                cluster_of[stem] = r["cluster"]
    clusters = {c: i for i, c in enumerate(sorted(set(cluster_of.values())))}
    return cluster_of, clusters


# ---------------------------------------------------------------- бутстрэп

def bootstrap(runs_t, runs_c, thr_t, thr_c, n_clusters, scfg, budget=None, rng=None):
    """runs_t / runs_c — списки Run по seed (в одном порядке: пары). Возвращает точечные оценки и реплики Δ.

    Δ в реплике = среднее по выбранным с возвращением индексам seed разностей treatment − control
    на выбранных с возвращением кластерах.
    """
    b = scfg["bootstrap"]
    S = len(runs_t)
    B = b["replicas"]
    rng = rng or np.random.default_rng(b["seed"])
    seed_idx = rng.integers(0, S, size=(B, S)) if b["resample_seeds"] else np.tile(np.arange(S), (B, 1))
    keys = ["recall", "fppi", "ap"] + (["recall_budget"] if budget is not None else [])
    reps = {k: np.empty((2, S, B)) for k in keys}
    p = np.full(n_clusters, 1.0 / n_clusters)
    for start in range(0, B, b["chunk"]):
        k = min(b["chunk"], B - start)
        W = rng.multinomial(n_clusters, p, size=k).astype(float)
        for g, (runs, thrs) in enumerate(((runs_t, thr_t), (runs_c, thr_c))):
            for s in range(S):
                m = runs[s].metrics(W, thrs[s], budget)
                for key in keys:
                    reps[key][g, s, start:start + k] = m[key]
    ones = np.ones((1, n_clusters))
    point = {key: np.empty((2, S)) for key in keys}
    for g, (runs, thrs) in enumerate(((runs_t, thr_t), (runs_c, thr_c))):
        for s in range(S):
            m = runs[s].metrics(ones, thrs[s], budget)
            for key in keys:
                point[key][g, s] = m[key][0]
    cols = np.arange(B)
    delta = {key: (reps[key][0][seed_idx, cols[:, None]] - reps[key][1][seed_idx, cols[:, None]]).mean(axis=1)
             for key in keys}
    est = {key: float((point[key][0] - point[key][1]).mean()) for key in keys}
    means = {key: point[key].mean(axis=1) for key in keys}       # [treatment, control], среднее по seed
    return est, delta, means


def percentile_ci(x, level):
    a = (1 - level) / 2
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan")
    return float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


# ---------------------------------------------------------------- правило решения

ESTABLISHED = ("established", "practically_significant")


def classify(lo, hi, margin):
    """Решение по скорректированному интервалу Δ_R (protocol.yaml, stats)."""
    if lo > margin:
        return "practically_significant"
    if lo > 0:
        return "established"
    if hi < 0:
        return "worse"
    return "not_established"


def decide(rows, scfg, pcfg):
    """rows — проверки одного сравнения treatment vs control. Возвращает решение по каждому датасету.

    Основной датасет (primary_datasets): гипотеза подтверждена, если условие mAP выполнено (когда оно обязательное)
    и хотя бы на одном уровне из hypothesis_levels — practically_significant. Вторичный — те же величины
    с пометкой «описательно», гипотеза по нему не проверяется.
    """
    res = {}
    binding = scfg["map"].get("binding", True)
    for ds in scfg["datasets"]:
        rec = sorted((r for r in rows if r["dataset"] == ds and r["metric"] == "recall"), key=lambda r: r["level"])
        mp = next((r for r in rows if r["dataset"] == ds and r["metric"] == "ap"), None)
        established = [r["level"] for r in rec if r["decision"] in ESTABLISHED]
        practical = [r["level"] for r in rec if r["decision"] == "practically_significant"
                     and r["value_control"] >= scfg["min_baseline_recall"]]
        measurable = [r["level"] for r in rec if r["value_control"] >= scfg["min_baseline_recall"]]
        # «порог полезности» — наименьший ИЗМЕРИМЫЙ уровень сетки, начиная с которого эффект established на нём
        # и на всех бо́льших измеримых уровнях; это точка сетки, а не непрерывный порог (см. text/stats.md)
        thr_level = next((L for L in measurable if all(x in established for x in measurable if x >= L)), None)
        map_ok = bool(mp and mp["decision"] == "non_inferior")
        primary = ds in scfg["primary_datasets"]
        d = {"role": "primary" if primary else "secondary (descriptive)",
             "map_ok": map_ok, "map_binding": binding,
             "established_levels": established, "practically_significant_levels": practical,
             "unmeasurable_levels": [r["level"] for r in rec if r["level"] not in measurable],
             "usefulness_level": thr_level}
        hyp = (map_ok or not binding) and any(L in practical for L in scfg["hypothesis_levels"])
        d["hypothesis_confirmed"] = hyp if primary else None
        res[ds] = d
    return res


def load_runs(root, dataset, part, regime, level, subset, seeds, thresholds, cluster_of, clusters):
    runs = []
    for seed in seeds:
        key = f"{dataset}__{regime}__s{seed}"
        meta = {"dataset": dataset, "split": part, "regime": regime, "seed": seed, "level": level,
                "subset": "not_occluded" if subset == "not_occluded" else "all",
                "population": "per_level" if subset == "all-per_level" else "all_levels",
                "fill": "noise" if subset == "all-noise" else "bg", "threshold": thresholds[key]}
        runs.append(Run(root / f"{key}__L{level}__{subset}.npz", cluster_of, clusters, meta))
    return runs, [thresholds[f"{dataset}__{regime}__s{seed}"] for seed in seeds]


def analyse(runs_dir, pcfg, scfg_override=None, tag="main", report_dir="reports", splits_dir="splits",
            treatment=None, control=None):
    scfg = {**pcfg["stats"], **(scfg_override or {})}
    treatment = treatment or scfg["treatment"]
    control = control or scfg["control"]
    part = scfg.get("split", "test")
    if (part != "test" or not scfg.get("independent_test", False)
            or scfg["subset"] != pcfg["stats"]["subset"]
            or (treatment, control) != (scfg["treatment"], scfg["control"])):
        scfg["primary_datasets"] = []
    runs_dir = Path(runs_dir)
    thresholds = json.loads((runs_dir / "thresholds.json").read_text(encoding="utf-8"))
    alpha, m = scfg["alpha"], scfg["bonferroni_m"]
    tol = pcfg["fppi_rule"]["fppi_tolerance"]
    rows = []
    rng = np.random.default_rng(scfg["bootstrap"]["seed"])
    for ds in scfg["datasets"]:
        first = runs_dir / f"{ds}__{treatment}__s{scfg['seeds'][0]}__L{scfg['recall_levels'][0]}__{scfg['subset']}.npz"
        evaluated = {str(i) for i in np.load(first)["images"]}
        cluster_of, clusters = load_clusters(Path(splits_dir) / f"{ds}_clusters.csv", evaluated,
                                             scfg.get("split", "test"))

        checks = [("recall", L) for L in scfg["recall_levels"]] + [("ap", scfg["map"]["level"])]
        for metric, level in checks:
            # AP при 0 % — по всем целям test (map.subset), Recall — по основной совокупности (subset)
            sub = scfg["map"].get("subset", scfg["subset"]) if metric == "ap" and scfg["subset"] == "all" \
                else scfg["subset"]
            rt, thr_t = load_runs(runs_dir, ds, part, treatment, level, sub, scfg["seeds"], thresholds,
                                  cluster_of, clusters)
            rc, thr_c = load_runs(runs_dir, ds, part, control, level, sub, scfg["seeds"], thresholds,
                                  cluster_of, clusters)
            if any(run.target_boxes != rt[0].target_boxes for run in rt + rc):
                raise ValueError(f"{ds}, L={level}, {sub}: различается состав целей")
            est, delta, means = bootstrap(rt, rc, thr_t, thr_c, len(clusters), scfg, rng=rng)
            row = {"dataset": ds, "metric": metric, "level": level, "treatment": treatment, "control": control,
                   "value_treatment": float(means[metric][0]), "value_control": float(means[metric][1]),
                   "estimate": est[metric], "fppi_treatment": float(means["fppi"][0]),
                   "fppi_control": float(means["fppi"][1])}
            invalid = ~np.isfinite(delta[metric])
            if metric == "recall":
                lvl = 1 - alpha / m
                lo, hi = percentile_ci(delta["recall"], lvl)
                row.update(ci_level=lvl, ci_low=lo, ci_high=hi)
                gap = abs(row["fppi_treatment"] - row["fppi_control"])
                row["fppi_gap"] = gap
                if gap > tol:
                    # правило разного FPPI: та же проверка при равном бюджете max(FPPI_t, FPPI_c)
                    budget = max(row["fppi_treatment"], row["fppi_control"])
                    est_b, delta_b, _ = bootstrap(rt, rc, thr_t, thr_c, len(clusters), scfg, budget=budget, rng=rng)
                    lo_b, hi_b = percentile_ci(delta_b["recall_budget"], lvl)
                    row["invalid_budget_replicas"] = int((~np.isfinite(delta_b["recall_budget"])).sum())
                    row.update(budget=budget, estimate_budget=est_b["recall_budget"], ci_low_budget=lo_b,
                               ci_high_budget=hi_b)
                    # преимущество засчитывается, только если сохраняется и при равном бюджете FPPI
                    lo = min(lo, lo_b)
                row["decision"] = classify(lo, hi, scfg["practical_margin"])
                row["descriptive"] = ds not in scfg["primary_datasets"]
            else:
                lvl = 1 - 2 * alpha              # двусторонний (1 − 2α) = односторонняя нижняя граница (1 − α)
                lo, hi = percentile_ci(delta["ap"], lvl)
                row.update(ci_level=1 - alpha, ci_low=lo, ci_high=float("inf"))
                row["decision"] = "non_inferior" if lo > -scfg["map"]["margin"] else "not_established"
                row["descriptive"] = ds not in scfg["primary_datasets"] or not scfg["map"].get("binding", True)
            row["invalid_replicas"] = int(invalid.sum())
            if (max(row["invalid_replicas"], row.get("invalid_budget_replicas", 0)) > len(invalid) * .01
                    or not np.isfinite(row["ci_low"])):
                row["decision"] = "unstable"
            rows.append(row)
    decision = decide(rows, scfg, pcfg)
    write_reports(rows, decision, scfg, tag, Path(report_dir))
    return rows, decision


def factorial_report(root, out, pcfg, scfg, datasets, part, controls, splits_dir="splits"):
    """Описательные контрасты по общим перевыборкам seed и групп кадров."""
    root = Path(root)
    design = pcfg["factorial"]
    thresholds = json.loads((root / "thresholds.json").read_text(encoding="utf-8"))
    seeds = scfg["seeds"]
    contrasts = {
        "fill_with_decoys": {"noise_decoy": 1, "bg_decoy": -1},
        "fill_without_decoys": {"noise_nodecoy": 1, "bg_nodecoy": -1},
        "decoy_with_bg": {"bg_decoy": 1, "bg_nodecoy": -1},
        "decoy_with_noise": {"noise_decoy": 1, "noise_nodecoy": -1},
        "interaction": {"noise_decoy": 1, "bg_decoy": -1, "noise_nodecoy": -1, "bg_nodecoy": 1},
    }
    rows = []
    ci_level = 1 - scfg["alpha"] / scfg["bonferroni_m"]
    for ds in datasets:
        for level in design["levels"]:
            first = root / f"{ds}__{design['regimes']['bg_decoy']}__s{seeds[0]}__L{level}__all.npz"
            with np.load(first) as z:
                images = set(map(str, z["images"]))
            cluster_of, clusters = load_clusters(Path(splits_dir) / f"{ds}_clusters.csv", images, part)
            draws = {}
            reference = None
            for cell, regime in design["regimes"].items():
                bg, thr = load_runs(root, ds, part, regime, level, "all", seeds, thresholds, cluster_of, clusters)
                noise, _ = load_runs(root, ds, part, regime, level, "all-noise", seeds, thresholds, cluster_of, clusters)
                if reference is None:
                    reference = bg[0].target_boxes
                if any(run.target_boxes != reference for run in bg + noise):
                    raise ValueError(f"{ds}, L={level}: совокупности схемы 2×2 различаются")
                # Нулевой детектор даёт базу Recall=0 на тех же целях.
                estimate, delta, _ = bootstrap(noise, bg, thr, thr, len(clusters), scfg)
                bg_est, bg_delta, _ = bootstrap(bg, bg, thr, [float("inf")] * len(seeds), len(clusters), scfg)
                draws[cell] = {"leak_index": (estimate["recall"], delta["recall"]),
                               "recall_bg": (bg_est["recall"], bg_delta["recall"])}
                fill = cell.split("_")[0]
                patch = [r for r in controls if r["dataset"] == ds and r["split"] == part
                         and r["regime"] == regime and r["fill"] == fill]
                complete = (len(patch) == len(seeds) and {r["seed"] for r in patch} == set(seeds)
                            and all(0 <= r["patches_hit"] <= r["patches"] and r["patches"] > 0 for r in patch))
                patch_rate = float(np.mean([r["patches_hit"] / r["patches"] for r in patch])) if complete else None
                for metric, (value, replicas) in draws[cell].items():
                    low, high = percentile_ci(replicas, ci_level)
                    rows.append([ds, part, level, cell, metric, value, low, high,
                                 int((~np.isfinite(replicas)).sum()), patch_rate])
            for name, weights in contrasts.items():
                for metric in ("leak_index", "recall_bg"):
                    value = sum(weight * draws[cell][metric][0] for cell, weight in weights.items())
                    replicas = sum(weight * draws[cell][metric][1] for cell, weight in weights.items())
                    low, high = percentile_ci(replicas, ci_level)
                    rows.append([ds, part, level, name, metric, value, low, high,
                                 int((~np.isfinite(replicas)).sum()), None])
    with (Path(out) / "factorial.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "split", "level", "effect", "metric", "estimate", "ci_low", "ci_high",
                         "invalid_replicas", "patch_hit_rate"])
        writer.writerows(rows)


def write_reports(rows, decision, scfg, tag, rep):
    rep.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with open(rep / f"stats_{tag}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    b = scfg["bootstrap"]
    lines = [f"# Статистика: {rows[0]['treatment']} против {rows[0]['control']} ({tag})", "",
             f"Бутстрэп: {b['replicas']} реплик, группы части {scfg.get('split', 'test')} + seed {scfg['seeds']} "
             f"({'перевыбираются' if b['resample_seeds'] else 'фиксированы'}), интервал `{scfg['interval']}`. "
             f"Recall: двусторонний {1 - scfg['alpha'] / scfg['bonferroni_m']:.4%} (Бонферрони, m = {scfg['bonferroni_m']} "
             f"уровней основного датасета {', '.join(scfg['primary_datasets'])}); practically_significant — нижняя "
             f"граница > {scfg['practical_margin']}. AP@0.5: односторонняя нижняя граница {1 - scfg['alpha']:.0%}, "
             f"допуск −{scfg['map']['margin']}"
             + ("" if scfg["map"].get("binding", True) else " (условие описательное, в правило не входит)") + ". "
             f"Вторичные датасеты ({', '.join(scfg['secondary_datasets'])}) — описательно, в итоговое правило не входят.",
             "",
             "| датасет | метрика | уровень | treatment | control | Δ | интервал | FPPI t / c | решение |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = f"[{r['ci_low']:+.3f}; {r['ci_high']:+.3f}]" if np.isfinite(r["ci_high"]) else f"[{r['ci_low']:+.3f}; +∞)"
        if "budget" in r:
            ci += f"; при FPPI {r['budget']:.3f}: [{r['ci_low_budget']:+.3f}; {r['ci_high_budget']:+.3f}]"
        mark = " (описательно)" if r.get("descriptive") else ""
        mark += f"; невалидных реплик: {r['invalid_replicas']}"
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['level']:.0%} | {r['value_treatment']:.3f} | "
                     f"{r['value_control']:.3f} | {r['estimate']:+.3f} | {ci} | "
                     f"{r['fppi_treatment']:.3f} / {r['fppi_control']:.3f} | {r['decision']}{mark} |")
    lines += ["", "## Решение", ""]
    pct = lambda xs: ", ".join(f"{x:.0%}" for x in xs) or "нет"
    for ds, d in decision.items():
        ul = f"{d['usefulness_level']:.0%}" if d["usefulness_level"] is not None else "не установлен"
        lines.append(f"- **{ds}** ({d['role']}): условие mAP {'выполнено' if d['map_ok'] else 'НЕ выполнено'}"
                     f"{'' if d['map_binding'] else ' (описательно)'}; established: {pct(d['established_levels'])}; "
                     f"practically_significant: {pct(d['practically_significant_levels'])}; "
                     f"неизмеримые уровни (Recall baseline < {scfg['min_baseline_recall']}): "
                     f"{pct(d['unmeasurable_levels'])}; порог полезности (точка сетки, по измеримым уровням): {ul}; "
                     "окончательное решение формируется вместе с контролем заплаток.")
    (rep / f"stats_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="каталог с .npz и thresholds.json")
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--replicas", type=int, help="переопределить число реплик (только для smoke test)")
    ap.add_argument("--treatment")
    ap.add_argument("--control")
    args = ap.parse_args()
    with open(args.protocol, encoding="utf-8") as fh:
        pcfg = yaml.safe_load(fh)
    override = {}
    if args.replicas:
        override["bootstrap"] = {**pcfg["stats"]["bootstrap"], "replicas": args.replicas}
    rows, decision = analyse(args.runs, pcfg, override, args.tag, treatment=args.treatment, control=args.control)
    for r in rows:
        print(f"{r['dataset']:5s} {r['metric']:6s} {r['level']:.0%} Δ={r['estimate']:+.4f} "
              f"[{r['ci_low']:+.4f}; {r['ci_high']:+.4f}] {r['decision']}")


if __name__ == "__main__":
    main()
