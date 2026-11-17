import math
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate import (Evaluation, compare_fppi_rule, covered_fraction, evaluate,  # noqa: E402
                      level_ground_truth, match_image, nms)

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8"))

T = (100, 100, 140, 180)           # целевая рамка 40x80


def shifted(box, dx=0, dy=0):
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


# ---------------------------------------------------------------- NMS

def test_nms_suppresses_overlap_keeps_separate():
    dets = [(*T, 0.9), (*shifted(T, 2), 0.8), (*shifted(T, 200), 0.7)]
    assert [d[4] for d in nms(dets, CFG)] == [0.9, 0.7]


def test_nms_conf_and_max_det():
    cfg = {**CFG, "nms": {**CFG["nms"], "max_det": 2}}
    dets = [(*shifted(T, 100 * i), s) for i, s in enumerate([0.0005, 0.5, 0.6, 0.7])]
    kept = nms(dets, cfg)
    assert [d[4] for d in kept] == [0.7, 0.6]               # 0.0005 < conf, третья — сверх max_det


# ---------------------------------------------------------------- сопоставление

def test_perfect_detection_is_tp():
    out, taken = match_image([(*T, 0.9)], [T], [], CFG)
    assert out == [(0.9, "tp")] and taken == {0}


def test_duplicate_is_fp_and_higher_confidence_wins():
    out, _ = match_image([(*shifted(T, 3), 0.6), (*T, 0.9)], [T], [], CFG)
    assert out == [(0.9, "tp"), (0.6, "fp")]


def test_detection_in_ignore_region_is_ignored():
    ign = (300, 100, 360, 200)
    out, _ = match_image([(310, 110, 350, 190, 0.8)], [], [ign], CFG)
    assert out == [(0.8, "ign")]


def test_gt_matched_before_ignore():
    # детекция совпадает и с целью, и с ignore-областью -> TP (сначала цели)
    out, _ = match_image([(*T, 0.8)], [T], [T], CFG)
    assert out == [(0.8, "tp")]


def test_duplicate_inside_ignore_still_fp():
    # дубль найденной цели — FP, даже если лежит в ignore (дубль проверяется раньше ignore)
    out, _ = match_image([(*T, 0.9), (*T, 0.8)], [T], [T], CFG)
    assert out == [(0.9, "tp"), (0.8, "fp")]


def test_low_iou_is_fp():
    out, _ = match_image([(*shifted(T, 30), 0.9)], [T], [], CFG)   # IoU = 10/70 < 0.5
    assert out == [(0.9, "fp")]


def test_detection_goes_to_best_free_target():
    t2 = shifted(T, 10)
    out, taken = match_image([(*t2, 0.9), (*T, 0.8)], [T, t2], [], CFG)
    assert out == [(0.9, "tp"), (0.8, "tp")] and taken == {0, 1}


# ---------------------------------------------------------------- метрики

def toy_eval():
    """4 изображения, 2 цели. Детекции: TP 0.9, FP 0.8, TP 0.7, FP 0.6."""
    gt = {"a": ([T], []), "b": ([T], []), "c": ([], []), "d": ([], [])}
    dets = {"a": [(*T, 0.9)], "c": [(*T, 0.8)], "b": [(*T, 0.7)], "d": [(*T, 0.6)]}
    ev, _ = evaluate(dets, gt, ["a", "b", "c", "d"], CFG)
    return ev


def test_fppi_counts_all_images():
    ev = toy_eval()
    assert ev.at_threshold(0.0) == {"recall": 1.0, "fppi": 0.5, "tp": 2, "fp": 2}
    assert ev.at_threshold(0.75) == {"recall": 0.5, "fppi": 0.25, "tp": 1, "fp": 1}


def test_threshold_for_fppi_is_lowest_admissible():
    ev = toy_eval()
    assert ev.threshold_for_fppi(0.25) == 0.7               # 0.7: FP=1 -> FPPI 0.25; 0.6 даёт 0.5
    assert ev.threshold_for_fppi(0.1) == 0.9                # только первая детекция
    assert ev.recall_at_fppi(0.25) == 1.0


def test_threshold_ties_are_grouped():
    gt = {"a": ([T], []), "b": ([], [])}
    dets = {"a": [(*T, 0.5)], "b": [(*T, 0.5)]}             # TP и FP с одинаковым score
    ev, _ = evaluate(dets, gt, ["a", "b"], CFG)
    assert ev.threshold_for_fppi(0.1) == math.inf           # порог 0.5 впускает и FP — допустимого нет


def test_ap_by_hand():
    # precision: 1, 1/2, 2/3, 2/4; recall: .5, .5, 1, 1 -> огибающая: 1 на [0,.5], 2/3 на [.5,1]
    assert toy_eval().ap() == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_ap_perfect_and_empty():
    ev, _ = evaluate({"a": [(*T, 0.9)]}, {"a": ([T], [])}, ["a"], CFG)
    assert ev.ap() == 1.0
    ev, _ = evaluate({}, {"a": ([T], [])}, ["a"], CFG)
    assert ev.ap() == 0.0


# ---------------------------------------------------------------- уровни масок и подмножества

def test_covered_fraction_union_not_sum():
    box = (0, 0, 10, 10)
    assert covered_fraction(box, [(0, 0, 5, 10), (0, 0, 5, 10)]) == 0.5
    assert covered_fraction(box, [(0, 0, 5, 10), (5, 0, 10, 5)]) == 0.75


def test_neighbor_mask_moves_target_to_ignore():
    a = {"id": "a", "box": (0, 0, 10, 10), "occluded": False}
    b = {"id": "b", "box": (5, 0, 15, 10), "occluded": False}
    masks = {"a": (2, 0, 9, 10), "b": (12, 0, 15, 10)}       # маска a закрывает 40 % рамки b
    keep, ign, ids = level_ground_truth([a, b], [], masks, CFG)
    assert ids == ["a"] and b["box"] in ign


def test_subset_moves_occluded_to_ignore():
    a = {"id": "a", "box": T, "occluded": True}
    keep, ign, ids = level_ground_truth([a], [], {}, CFG, lambda t: not t["occluded"])
    assert keep == [] and ign == [T]


# ---------------------------------------------------------------- правило разного FPPI

def make_ev(n_images, n_targets, tps, fps):
    ev = Evaluation(n_images, n_targets)
    ev.add([(s, "tp") for s in tps] + [(s, "fp") for s in fps])
    return ev


def test_fppi_rule_close_fppi_uses_primary():
    a = make_ev(100, 10, [0.9] * 6, [0.5] * 5)
    b = make_ev(100, 10, [0.9] * 5, [0.5] * 5)
    r = compare_fppi_rule(a, b, 0.4, 0.4, CFG)
    assert not r["budget_matched"] and r["verdict"] == "a"


def test_fppi_rule_large_gap_reversal_not_established():
    # A выигрывает при пороге только за счёт большого FPPI; при равном бюджете B лучше
    # A: при пороге 0.25 Recall 0.7, FPPI 0.15. B: при пороге 0.25 Recall 0.6, FPPI 0,
    # но ниже порога у B есть ещё 3 TP, и при бюджете 0.15 его Recall = 0.9.
    a = make_ev(100, 10, [0.9] * 4 + [0.3] * 3, [0.8] * 15)
    b = make_ev(100, 10, [0.9] * 6 + [0.1] * 3, [0.2] * 5)
    r = compare_fppi_rule(a, b, 0.25, 0.25, CFG)
    assert r["fppi_gap"] > CFG["fppi_rule"]["fppi_tolerance"]
    assert r["recall_a"] > r["recall_b"] and r["verdict"] == "not_established"


def test_fppi_rule_large_gap_confirmed():
    a = make_ev(100, 10, [0.9] * 8, [0.5] * 10)
    b = make_ev(100, 10, [0.9] * 5, [0.5] * 2 + [0.1] * 20)
    r = compare_fppi_rule(a, b, 0.4, 0.4, CFG)
    assert r["budget_matched"] and r["verdict"] == "a"
