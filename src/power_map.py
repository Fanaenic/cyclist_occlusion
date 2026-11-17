"""Синтетика ДО обучения (fix_plan, шаг 5): какая нижняя граница ΔAP получается у ОДИНАКОВЫХ моделей
на реальной структуре test BDD100K — те же изображения, кластеры и число целей, та же процедура src/stats.py.

Модель детекций (одинаковая для proposed и baseline, 3 seed каждого):
  - у каждой цели — общая для всех моделей трудность d_t ~ N(0, 1); score детекции цели моделью
    = sigmoid(mu_tp + sqrt(rho)·d_t + sqrt(1 − rho)·e), e ~ N(0, 1) — своя для модели;
  - у каждого изображения — общая «трудность фона» v_i ~ N(0, 1); число ложных детекций модели
    ~ Poisson(fp_rate · exp(0.5·v_i)), score = sigmoid(N(mu_fp, 1)).
rho — сходство моделей между собой (насколько одни и те же цели трудны для всех).

Сценарии заданы в configs/protocol.yaml (map_power) до запуска; основной — central.
Выход: reports/map_power.csv, reports/map_power.md.

Запуск: python src/power_map.py [--protocol configs/protocol.yaml]
"""
import argparse
import csv
import tempfile
import zlib
from pathlib import Path

import numpy as np
import yaml

from evaluate import Evaluation
from stats import Run, bootstrap, load_clusters, percentile_ci


def test_structure(ycfg, ds):
    """Изображения test (как в datasets_yolo) и число целей на изображении."""
    root = Path(ycfg["out_dir"]) / ds / "labels" / "test"
    return {p.stem: sum(1 for ln in p.read_text().splitlines() if ln.strip()) for p in sorted(root.glob("*.txt"))}


def simulate_model(rng, n_targets, d, v, sc):
    sig = lambda x: 1 / (1 + np.exp(-x))
    ev = Evaluation(len(n_targets), sum(n_targets.values()))
    k = 0
    for i, (img, nt) in enumerate(n_targets.items()):
        m = [(float(sig(sc["mu_tp"] + np.sqrt(sc["rho"]) * d[k + j] + np.sqrt(1 - sc["rho"]) * rng.normal())), "tp")
             for j in range(nt)]
        k += nt
        m += [(float(sig(rng.normal(sc["mu_fp"], 1.0))), "fp")
              for _ in range(rng.poisson(sc["fp_rate"] * np.exp(0.5 * v[i])))]
        ev.add(m, img)
        ev.targets_per_image[img] = nt
    return ev


def run_scenario(name, sc, n_targets, cluster_of, clusters, pcfg, mp, tmp):
    scfg = {**pcfg["stats"], "bootstrap": {**pcfg["stats"]["bootstrap"], "replicas": mp["replicas"]}}
    S = len(scfg["seeds"])
    lows, aps, recs = [], [], []
    for rep in range(mp["repeats"]):
        rng = np.random.default_rng([mp["seed"], rep, zlib.crc32(name.encode())])
        d = rng.normal(size=sum(n_targets.values()))
        v = rng.normal(size=len(n_targets))
        runs = []
        for g in range(2 * S):
            ev = simulate_model(rng, n_targets, d, v, sc)
            path = Path(tmp) / f"{name}_{rep}_{g}.npz"
            ev.save(path)
            runs.append(Run(path, cluster_of, clusters))
            aps.append(ev.ap())
            recs.append(ev.recall_at_fppi(pcfg["metrics"]["target_fppi"]))
        _, delta, _ = bootstrap(runs[:S], runs[S:], [1.0] * S, [1.0] * S, len(clusters), scfg, rng=rng)
        lo, _ = percentile_ci(delta["ap"], 1 - 2 * scfg["alpha"])
        lows.append(lo)
    return {"scenario": name, **sc, "ap_mean": float(np.mean(aps)), "recall_at_fppi_mean": float(np.mean(recs)),
            "lower_bound_median": float(np.median(lows)), "lower_bound_min": float(np.min(lows)),
            "lower_bound_max": float(np.max(lows)), "repeats": mp["repeats"], "replicas": mp["replicas"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="configs/protocol.yaml")
    ap.add_argument("--yolo-config", default="configs/yolo.yaml")
    args = ap.parse_args()
    with open(args.protocol, encoding="utf-8") as fh:
        pcfg = yaml.safe_load(fh)
    with open(args.yolo_config, encoding="utf-8") as fh:
        ycfg = yaml.safe_load(fh)
    mp = pcfg["map_power"]
    ds = mp["dataset"]
    n_targets = test_structure(ycfg, ds)
    cluster_of, clusters = load_clusters(Path(ycfg["splits_dir"]) / f"{ds}_clusters.csv", set(n_targets))
    pos_clusters = len({cluster_of[i] for i, n in n_targets.items() if n})
    print(f"{ds} test: {len(n_targets)} изображений, {sum(n_targets.values())} целей, {len(clusters)} кластеров "
          f"({pos_clusters} с целями)")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, sc in mp["scenarios"].items():
            r = run_scenario(name, sc, n_targets, cluster_of, clusters, pcfg, mp, tmp)
            rows.append(r)
            print(f"{name:10s} AP≈{r['ap_mean']:.3f} Recall@FPPI≈{r['recall_at_fppi_mean']:.3f} "
                  f"нижняя граница ΔAP: медиана {r['lower_bound_median']:+.4f} "
                  f"[{r['lower_bound_min']:+.4f}; {r['lower_bound_max']:+.4f}]")
    rep = Path("reports")
    with open(rep / "map_power.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    margin = pcfg["stats"]["map"]["margin"]
    central = next(r for r in rows if r["scenario"] == mp["decision_scenario"])
    binding = central["lower_bound_median"] > -margin
    lines = ["# Условие mAP: ожидаемая нижняя граница ΔAP у одинаковых моделей (синтетика до обучения)", "",
             f"Структура test {ds}: {len(n_targets)} дневных изображений, {sum(n_targets.values())} целей, "
             f"{len(clusters)} кластеров ({pos_clusters} с целями). Та же процедура, что в src/stats.py: 3 + 3 модели, "
             f"двухступенчатый бутстрэп ({mp['replicas']} реплик), односторонняя нижняя граница "
             f"{1 - pcfg['stats']['alpha']:.0%}. Каждый сценарий повторён {mp['repeats']} раз (новые синтетические модели).",
             "", "| сценарий | rho | AP | Recall@FPPI=0,1 | нижняя граница ΔAP: медиана | мин | макс |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['scenario']} | {r['rho']} | {r['ap_mean']:.3f} | {r['recall_at_fppi_mean']:.3f} | "
              f"{r['lower_bound_median']:+.4f} | {r['lower_bound_min']:+.4f} | {r['lower_bound_max']:+.4f} |" for r in rows]
    lines += ["", f"Правило (fix_plan, шаг 5): если у одинаковых моделей нижняя граница хуже −{margin}, условие mAP "
              "переводится в описательное. Решающий сценарий, заданный до запуска: "
              f"`{mp['decision_scenario']}` — медиана нижней границы {central['lower_bound_median']:+.4f}.", "",
              "**Итог:** " + ("условие mAP остаётся обязательным." if binding else
                              "условие mAP переводится в описательное (отчитывается, в правило не входит).")]
    (rep / "map_power.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-1:]))


if __name__ == "__main__":
    main()
