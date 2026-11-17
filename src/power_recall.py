"""Синтетический расчёт мощности для Recall на test основного датасета ДО обучения (по аналогии с power_map.py).

Вопрос: при каком истинном приросте Recall (proposed − baseline) вероятность не ниже target_power, что нижняя
граница скорректированного интервала (98,75 %, двухступенчатый бутстрэп, как в src/stats.py) окажется
  а) > 0 (решение established),
  б) > practical_margin (решение practically_significant).

Структура — реальная: изображения test, кластеры и цели основной совокупности (валидные на всех уровнях).
Модель попаданий (probit): цель t найдена моделью m при пороге, если sqrt(rho)·d_t + sqrt(1 − rho)·e_mt > −z_m,
где d_t ~ N(0, 1) — общая для всех моделей трудность цели, e_mt ~ N(0, 1), z_m = Φ⁻¹(r_m) — так P(найдена) = r_m.
Recall модели r_m = r + N(0, seed_sd): seed_sd — разброс от обучения с разными seed; у baseline r = r_b, у proposed
r = r_b + Δ. FPPI моделей считаются равными (правило разного FPPI не моделируется).
Для разных Δ используются одни и те же случайные числа (общие d, e, веса кластеров) — кривая мощности гладкая.

Параметры — configs/protocol.yaml (recall_power). Выход: reports/recall_power.csv, reports/recall_power.md.
Запуск: python src/power_recall.py
"""
import argparse
import csv
from pathlib import Path
from statistics import NormalDist

import numpy as np
import yaml

from evaluate import load_gt, valid_on_all_levels
from stats import load_clusters


def structure(ds, pcfg, ycfg):
    """Число целей основной совокупности по кластерам test (только кластеры с целями) и число всех кластеров."""
    images, targets, _, masks = load_gt(ds, "test", ycfg)
    valid = valid_on_all_levels(targets, masks, pcfg)
    cluster_of, clusters = load_clusters(Path(ycfg["splits_dir"]) / f"{ds}_clusters.csv", set(images))
    per_target_cluster = [clusters[cluster_of[img]] for img, ts in targets.items() for t in ts if t["id"] in valid]
    return np.array(per_target_cluster), len(clusters), len(images)


def simulate(sc, deltas, tcl, n_clusters, scfg, rp, rng):
    """Для одного повтора: нижние границы интервала Δ_R при каждом Δ (общие случайные числа для всех Δ)."""
    S, B = len(scfg["seeds"]), rp["replicas"]
    T = len(tcl)
    nd = NormalDist()
    d = rng.normal(size=T)
    e = rng.normal(size=(2, S, T))
    r_jit = rng.normal(0, sc["seed_sd"], size=(2, S))
    W = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=B).astype(float)
    pos = np.unique(tcl)
    Wp = W[:, pos]                                         # веса кластеров с целями
    col = np.searchsorted(pos, tcl)
    nt = np.bincount(col, minlength=len(pos)).astype(float)
    denom = Wp @ nt                                        # число целей в реплике
    seed_idx = rng.integers(0, S, size=(B, S)) if scfg["bootstrap"]["resample_seeds"] else np.tile(np.arange(S), (B, 1))
    latent = np.sqrt(sc["rho"]) * d + np.sqrt(1 - sc["rho"]) * e      # (2, S, T)
    q = (scfg["alpha"] / scfg["bonferroni_m"]) / 2
    lows = []
    for delta in deltas:
        rec = np.empty((2, S, B))
        for g, base in enumerate((sc["baseline_recall"] + delta, sc["baseline_recall"])):   # 0 — proposed, 1 — baseline
            for s in range(S):
                r = float(np.clip(base + r_jit[g, s], 1e-4, 1 - 1e-4))
                hit = latent[g, s] > -nd.inv_cdf(r)
                tp = np.bincount(col, weights=hit, minlength=len(pos))
                rec[g, s] = (Wp @ tp) / denom
        cols = np.arange(B)[:, None]
        delta_rep = (rec[0][seed_idx, cols] - rec[1][seed_idx, cols]).mean(axis=1)
        lows.append(np.quantile(delta_rep, q))
    return np.array(lows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--yolo-config", default="configs/yolo.yaml")
    args = ap.parse_args()
    with open(args.protocol, encoding="utf-8") as fh:
        pcfg = yaml.safe_load(fh)
    with open(args.yolo_config, encoding="utf-8") as fh:
        ycfg = yaml.safe_load(fh)
    rp, scfg = pcfg["recall_power"], pcfg["stats"]
    ds = rp["dataset"]
    tcl, n_clusters, n_images = structure(ds, pcfg, ycfg)
    deltas = np.round(np.arange(rp["delta_min"], rp["delta_max"] + 1e-9, rp["delta_step"]), 4)
    margin, target = scfg["practical_margin"], rp["target_power"]
    level = 1 - scfg["alpha"] / scfg["bonferroni_m"]
    print(f"{ds} test: {n_images} изображений, {n_clusters} кластеров, целей основной совокупности {len(tcl)} "
          f"в {len(np.unique(tcl))} кластерах; интервал {level:.4%}")
    rows, summary = [], []
    for name, sc in rp["scenarios"].items():
        rng = np.random.default_rng([rp["seed"], sum(map(ord, name))])
        lows = np.array([simulate(sc, deltas, tcl, n_clusters, scfg, rp, rng) for _ in range(rp["repeats"])])
        p0, pm = (lows > 0).mean(axis=0), (lows > margin).mean(axis=0)
        for dlt, a, b in zip(deltas, p0, pm):
            rows.append({"scenario": name, **sc, "delta": float(dlt), "power_lower_gt_0": float(a),
                         f"power_lower_gt_{margin}": float(b)})
        first = lambda p: next((float(x) for x, v in zip(deltas, p) if v >= target), None)
        summary.append((name, sc, first(p0), first(pm)))
        print(f"{name:12s} Δ для мощности {target:.0%}: нижняя граница > 0 — {first(p0)}, > {margin} — {first(pm)}")
    rep = Path("reports")
    with open(rep / "recall_power.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fmt = lambda x: f"+{x:.2f}" if x is not None else f"> +{rp['delta_max']:.2f}"
    lines = ["# Мощность проверки Recall (синтетика до обучения)", "",
             f"Test {ds}: {n_images} дневных изображений, {n_clusters} кластеров; цели основной совокупности — "
             f"{len(tcl)} в {len(np.unique(tcl))} кластерах. Процедура как в src/stats.py: 3 seed proposed + 3 seed "
             f"baseline, двухступенчатый бутстрэп ({rp['replicas']} реплик), двусторонний интервал {level:.2%} "
             f"(Бонферрони, m = {scfg['bonferroni_m']}). Каждый сценарий — {rp['repeats']} повторов. "
             "Модель попаданий и её параметры — в src/power_recall.py и configs/protocol.yaml (recall_power). "
             "FPPI моделей считаются равными.", "",
             f"## Истинный прирост Recall, при котором мощность ≥ {target:.0%}", "",
             "| сценарий | Recall baseline | rho | разброс между seed | нижняя граница > 0 | "
             f"нижняя граница > +{margin} |", "|---|---|---|---|---|---|"]
    lines += [f"| {n} | {sc['baseline_recall']} | {sc['rho']} | {sc['seed_sd']} | {fmt(a)} | {fmt(b)} |"
              for n, sc, a, b in summary]
    lines += ["", "## Кривые мощности (основной сценарий)", "",
              f"| Δ | P(нижняя граница > 0) | P(нижняя граница > +{margin}) |", "|---|---|---|"]
    lines += [f"| +{r['delta']:.2f} | {r['power_lower_gt_0']:.2f} | {r[f'power_lower_gt_{margin}']:.2f} |"
              for r in rows if r["scenario"] == rp["main_scenario"]]
    lines += ["", "Чтение: при истинном приросте меньше указанного в таблице проверка с большой вероятностью даст "
              "«не установлено» даже при реальном эффекте. Порог practically_significant требует, чтобы нижняя "
              f"граница превысила +{margin}, поэтому истинный эффект должен быть заметно больше +{margin}."]
    (rep / "recall_power.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
