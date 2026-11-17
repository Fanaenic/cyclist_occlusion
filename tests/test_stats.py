import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import Evaluation  # noqa: E402
from stats import Run, analyse, classify, decide  # noqa: E402

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
    return {"datasets": [ds], "primary_datasets": [ds], "secondary_datasets": [],
            "map": {**PCFG["stats"]["map"], "subset": "all"},          # в синтетике один вариант совокупности
            "bootstrap": {**PCFG["stats"]["bootstrap"], "replicas": 400, "chunk": 100}}


def test_clear_improvement_detected_identical_not(tmp_path):
    over = make_runs(tmp_path, 0.8, 0.8)
    rows, dec = analyse(tmp_path, PCFG, over, tag="t", report_dir=tmp_path, splits_dir=tmp_path)
    by = {(r["metric"], r["level"]): r for r in rows}
    # 0.8·0.73 против 0.8·0.46: разность ≈ 0.22 — практически значимое улучшение
    assert by[("recall", 0.9)]["decision"] == "practically_significant"
    assert by[("recall", 0.9)]["ci_low"] > PCFG["stats"]["practical_margin"]
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


SCFG = {**PCFG["stats"], "datasets": ["x"], "primary_datasets": ["x"], "secondary_datasets": [],
        "map": {**PCFG["stats"]["map"], "binding": True}}     # логику обязательного условия проверяем явно


def rec(level, decision, baseline=0.5):
    return {"dataset": "x", "metric": "recall", "level": level, "decision": decision, "value_control": baseline}


AP_OK = {"dataset": "x", "metric": "ap", "level": 0.0, "decision": "non_inferior"}


def test_classify_two_levels():
    m = PCFG["stats"]["practical_margin"]
    assert classify(m + 0.01, 0.3, m) == "practically_significant"
    assert classify(0.01, 0.3, m) == "established"
    assert classify(-0.2, -0.01, m) == "worse"
    assert classify(-0.01, 0.2, m) == "not_established"


def test_usefulness_level_is_contiguous_from_top():
    rows = [rec(0.3, "established"), rec(0.5, "not_established"), rec(0.7, "established"),
            rec(0.9, "practically_significant"), AP_OK]
    d = decide(rows, SCFG, PCFG)["x"]
    assert d["established_levels"] == [0.3, 0.7, 0.9] and d["usefulness_level"] == 0.7
    assert d["hypothesis_confirmed"]                                # 90 % — practically_significant


def test_hypothesis_needs_practical_effect_at_50_70_90():
    rows = [rec(0.3, "practically_significant"), rec(0.5, "established"), rec(0.7, "established"),
            rec(0.9, "established"), AP_OK]
    d = decide(rows, SCFG, PCFG)["x"]
    assert d["usefulness_level"] == 0.3 and not d["hypothesis_confirmed"]


def test_unmeasurable_level_excluded_from_usefulness():
    # на 90 % Recall baseline < 0.05: уровень неизмерим, порог считается по 30–70 %
    rows = [rec(0.3, "not_established"), rec(0.5, "established"), rec(0.7, "established"),
            rec(0.9, "not_established", baseline=0.01), AP_OK]
    d = decide(rows, SCFG, PCFG)["x"]
    assert d["unmeasurable_levels"] == [0.9] and d["usefulness_level"] == 0.5


def test_map_condition_blocks_hypothesis_unless_descriptive():
    rows = [rec(0.9, "practically_significant"),
            {"dataset": "x", "metric": "ap", "level": 0.0, "decision": "not_established"}]
    assert not decide(rows, SCFG, PCFG)["x"]["hypothesis_confirmed"]
    loose = {**SCFG, "map": {**SCFG["map"], "binding": False}}
    assert decide(rows, loose, PCFG)["x"]["hypothesis_confirmed"]


def test_secondary_dataset_is_descriptive():
    rows = [rec(0.9, "practically_significant"), AP_OK]
    d = decide(rows, {**SCFG, "primary_datasets": [], "secondary_datasets": ["x"]}, PCFG)["x"]
    assert d["hypothesis_confirmed"] is None and d["practically_significant_levels"] == [0.9]
