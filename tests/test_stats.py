import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import Evaluation  # noqa: E402
from stats import Run, analyse, decide  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PCFG = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))


def synthetic_eval(rng, n_clusters, imgs_per_cluster, p_hit, fp_rate):
    """Каждое изображение: 0–2 цели; цель найдена с вероятностью p_hit (score ~ U(0.3, 1)); FP ~ Poisson."""
    images = [f"c{c}_i{i}" for c in range(n_clusters) for i in range(imgs_per_cluster)]
    nt = {img: int(rng.integers(0, 3)) for img in images}
    ev = Evaluation(len(images), sum(nt.values()))
    for img in images:
        m = [(float(rng.uniform(0.3, 1.0)), "tp") for _ in range(nt[img]) if rng.random() < p_hit]
        m += [(float(rng.uniform(0.0, 0.8)), "fp") for _ in range(rng.poisson(fp_rate))]
        ev.add(m, img)
        ev.targets_per_image[img] = nt[img]
    return ev, {img: img.split("_")[0] for img in images}


def test_weighted_metrics_equal_evaluation(tmp_path):
    ev, cl = synthetic_eval(np.random.default_rng(0), 20, 10, 0.7, 0.3)
    ev.save(tmp_path / "r.npz")
    clusters = {c: i for i, c in enumerate(sorted(set(cl.values())))}
    run = Run(tmp_path / "r.npz", cl, clusters)
    m = run.metrics(np.ones((1, len(clusters))), 0.5, budget=0.1)
    ref = ev.at_threshold(0.5)
    assert m["recall"][0] == pytest.approx(ref["recall"])
    assert m["fppi"][0] == pytest.approx(ref["fppi"])
    assert m["ap"][0] == pytest.approx(ev.ap())
    assert m["recall_budget"][0] == pytest.approx(ev.recall_at_fppi(0.1))


def test_cluster_weights_duplicate_cluster(tmp_path):
    """Вес 2 у кластера = дважды взятый кластер: Recall совпадает с оценкой на «удвоенных» данных."""
    ev, cl = synthetic_eval(np.random.default_rng(1), 4, 5, 0.6, 0.2)
    ev.save(tmp_path / "r.npz")
    clusters = {c: i for i, c in enumerate(sorted(set(cl.values())))}
    run = Run(tmp_path / "r.npz", cl, clusters)
    W = np.array([[2.0, 0.0, 1.0, 1.0]])
    tp = sum(1 for s, t, i in zip(ev.scores, ev.labels, ev.det_images) if t and s >= 0.5
             for _ in range(int(W[0, clusters[cl[i]]])))
    nt = sum(ev.targets_per_image[i] * W[0, clusters[cl[i]]] for i in ev.targets_per_image)
    assert run.metrics(W, 0.5)["recall"][0] == pytest.approx(tp / nt)


def make_runs(tmp_path, p_hit_t, p_hit_c, n_clusters=60, seeds=(1, 2, 3)):
    rng = np.random.default_rng(42)
    ds = "toy"
    thr = {}
    cl = None
    for regime, p in (("proposed", p_hit_t), ("baseline", p_hit_c)):
        for s in seeds:
            thr[f"{ds}__{regime}__s{s}"] = 0.5
            for L in (0.0, 0.3, 0.5, 0.7, 0.9):
                # чем выше уровень, тем ниже вероятность найти цель; proposed теряет меньше
                ev, cl = synthetic_eval(np.random.default_rng(rng.integers(1 << 30)), n_clusters, 8,
                                        p * (1 - 0.6 * L) if regime == "baseline" else p * (1 - 0.3 * L), 0.2)
                ev.save(tmp_path / f"{ds}__{regime}__s{s}__L{L}__all.npz")
    (tmp_path / "thresholds.json").write_text(json.dumps(thr))
    with open(tmp_path / f"{ds}_clusters.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "cluster", "part"])
        w.writerows([(img, c, "test") for img, c in cl.items()])
    return {"datasets": [ds], "bootstrap": {**PCFG["stats"]["bootstrap"], "replicas": 400, "chunk": 100}}


def test_clear_improvement_detected_identical_not(tmp_path):
    over = make_runs(tmp_path, 0.8, 0.8)
    rows, dec = analyse(tmp_path, PCFG, over, tag="t", report_dir=tmp_path, splits_dir=tmp_path)
    by = {(r["metric"], r["level"]): r for r in rows}
    assert by[("recall", 0.9)]["decision"] == "improvement"          # 0.8·0.73 против 0.8·0.46
    assert by[("recall", 0.9)]["ci_low"] > 0
    # на 0 % режимы одинаковы по построению, но выборки независимы: ΔAP ≈ 0, интервал — шириной в шум
    assert abs(by[("ap", 0.0)]["estimate"]) < 0.02
    assert (tmp_path / "stats_t.md").exists()


def test_no_difference_not_established(tmp_path):
    # baseline теряет меньше, чем «proposed», только при уровне > 0: подменим — сравним proposed сам с собой
    over = make_runs(tmp_path, 0.8, 0.8)
    rows, _ = analyse(tmp_path, PCFG, over, tag="t", report_dir=tmp_path, splits_dir=tmp_path,
                      treatment="proposed", control="proposed")
    for r in rows:
        if r["metric"] == "recall":
            assert r["estimate"] == 0 and r["decision"] == "not_established"
            assert r["ci_low"] <= 0 <= r["ci_high"]
        else:
            assert r["estimate"] == 0 and r["decision"] == "non_inferior"


def test_usefulness_level_is_contiguous_from_top():
    scfg = {"datasets": ["x"]}
    rows = [{"dataset": "x", "metric": "recall", "level": L, "decision": d}
            for L, d in ((0.3, "improvement"), (0.5, "not_established"), (0.7, "improvement"), (0.9, "improvement"))]
    rows.append({"dataset": "x", "metric": "ap", "level": 0.0, "decision": "non_inferior"})
    d = decide(rows, scfg, PCFG)["x"]
    assert d["improved_levels"] == [0.3, 0.7, 0.9] and d["usefulness_level"] == 0.7 and d["useful"]


def test_map_condition_blocks_usefulness():
    rows = [{"dataset": "x", "metric": "recall", "level": 0.9, "decision": "improvement"},
            {"dataset": "x", "metric": "ap", "level": 0.0, "decision": "not_established"}]
    assert not decide(rows, {"datasets": ["x"]}, PCFG)["x"]["useful"]
