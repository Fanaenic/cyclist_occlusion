import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from masks import (apply_mask, area, inter, make_mask, pixel_box, random_erasing_ore,  # noqa: E402
                   rng_for, sample_training_mask)

SHAPE = {"aspect_ratio": [0.5, 2.0], "position": "uniform"}
NB = {"policy": "avoid", "max_cover": 0.1, "max_tries": 200}


def test_pixel_box_covers_float_box():
    assert pixel_box(10.2, 5.9, 20.1, 30.0) == (10, 5, 21, 30)


@pytest.mark.parametrize("level", [0.3, 0.5, 0.7, 0.9])
def test_fraction_within_tolerance_and_inside(level):
    rng = np.random.default_rng(0)
    for _ in range(500):
        W, H = int(rng.integers(8, 200)), int(rng.integers(25, 300))
        box = (100, 50, 100 + W, 50 + H)
        m, _, _ = make_mask(box, level, rng, SHAPE)
        assert inter(m, box) == area(m)                        # целиком внутри рамки
        assert abs(area(m) / area(box) - level) <= 0.02        # ±2 п.п.


def test_level_zero_is_empty():
    m, cover, flag = make_mask((0, 0, 30, 60), 0.0, np.random.default_rng(0), SHAPE)
    assert area(m) == 0 and cover == 0 and not flag


def test_deterministic_by_key_not_order():
    a = make_mask((0, 0, 40, 80), 0.5, rng_for(1, "kitti", "img", "obj", 0.5), SHAPE)
    _ = make_mask((0, 0, 40, 80), 0.5, rng_for(1, "kitti", "img", "other", 0.5), SHAPE)
    b = make_mask((0, 0, 40, 80), 0.5, rng_for(1, "kitti", "img", "obj", 0.5), SHAPE)
    assert a == b


def test_avoids_neighbor_when_possible():
    box = (0, 0, 100, 100)
    neighbor = (0, 0, 50, 100)                                 # левая половина рамки
    m, cover, flag = make_mask(box, 0.3, np.random.default_rng(3), SHAPE, [neighbor], NB)
    assert cover <= 0.1 and not flag


def test_flags_unavoidable_neighbor():
    box = (0, 0, 100, 100)
    m, cover, flag = make_mask(box, 0.9, np.random.default_rng(3), SHAPE, [(10, 10, 90, 90)], NB)
    assert flag and cover > 0.1


def test_apply_mask_noise_reproducible():
    img = np.zeros((50, 50, 3), np.uint8)
    a = apply_mask(img, (10, 10, 20, 30), {"mode": "noise"}, 42)
    b = apply_mask(img, (10, 10, 20, 30), {"mode": "noise"}, 42)
    assert np.array_equal(a, b) and a[:10].sum() == 0 and a[10:30, 10:20].sum() > 0


def test_ore_inside_box_and_area_range():
    ore = {"p": 1.0, "area": [0.02, 0.4], "aspect_ratio": [0.3, 3.33], "max_tries": 100}
    rng = np.random.default_rng(0)
    box = (10, 10, 110, 210)
    for _ in range(300):
        m = random_erasing_ore(box, rng, ore)
        if m is None:
            continue
        assert inter(m, box) == area(m)
        assert 0.01 <= area(m) / area(box) <= 0.42


def test_proposed_levels_and_probability():
    regime = {"p": 0.5, "levels": [0.3, 0.5, 0.7], "weights": [1, 1, 1]}
    rng = np.random.default_rng(0)
    box = (0, 0, 60, 120)
    fracs = [sample_training_mask(box, rng, regime, SHAPE) for _ in range(2000)]
    applied = [area(m) / area(box) for m in fracs if m]
    assert 0.45 < len(applied) / len(fracs) < 0.55
    assert all(min(abs(f - lv) for lv in regime["levels"]) <= 0.02 for f in applied)


# ---------------------------------------------------------------- заливка фоном и приманки

from masks import apply_training_masks, pick_background, pick_like_targets  # noqa: E402


def test_pick_like_targets_stays_in_target_spans_and_avoids_boxes():
    rng = np.random.default_rng(0)
    avoid = [(0, 150, 300, 200)]
    spans = [[0.3, 0.6], [0.7, 0.9]]                    # диапазоны целей: 90..180 и 210..270 пикселей при H = 300
    for _ in range(200):
        x0, y0, clean = pick_like_targets(400, 300, 30, 40, avoid, rng, 200, spans)
        assert clean and inter((x0, y0, x0 + 30, y0 + 40), avoid[0]) == 0
        assert 90 <= y0 and y0 + 40 <= 180 or 210 <= y0 and y0 + 40 <= 270


def test_pick_like_targets_patch_taller_than_span_is_bottom_aligned():
    x0, y0, _ = pick_like_targets(400, 300, 20, 100, [], np.random.default_rng(0), 10, [[0.5, 0.6]])
    assert y0 + 100 == 180

BG_CFG = {
    "shape": SHAPE, "neighbors": NB, "fill": {"mode": "background"}, "background": {"max_tries": 200},
    "placement": {"mode": "like_targets"},
    "training_forbidden": {"part_of_target_ioa": 0.9},
    "training": {"proposed": {"p": 1.0, "levels": [0.5], "weights": [1], "decoys": True},
                 "ore": {"generator": "ore", "p": 1.0, "area": [0.1, 0.3],
                         "aspect_ratio": [0.5, 2.0], "max_tries": 100, "decoys": True}},
}


def test_pick_background_avoids_boxes():
    rng = np.random.default_rng(0)
    avoid = [(0, 0, 100, 100), (150, 0, 200, 100)]
    for _ in range(100):
        sx, sy, clean = pick_background(200, 100, 20, 20, avoid, rng, 200)
        assert clean and all(inter((sx, sy, sx + 20, sy + 20), a) == 0 for a in avoid)


def test_apply_mask_background_copies_source():
    img = np.arange(60 * 80 * 3, dtype=np.uint32).reshape(60, 80, 3).astype(np.uint8)
    out = apply_mask(img, (10, 10, 20, 30), {"mode": "background"}, 0, (50, 5))
    assert np.array_equal(out[10:30, 10:20], img[5:25, 50:60])
    assert np.array_equal(out[:10], img[:10])


@pytest.mark.parametrize("regime", ["proposed", "ore"])
def test_training_decoys_match_masks_and_stay_on_background(regime):
    img = np.random.default_rng(1).integers(0, 255, (300, 600, 3), dtype=np.uint8)
    boxes = [(100, 100, 140, 180), (300, 120, 330, 170)]
    spans = [[100 / 300, 180 / 300], [120 / 300, 170 / 300]]                # диапазоны целей, доли высоты
    out, masks, decoys, skips = apply_training_masks(img, boxes, np.random.default_rng(2), regime, BG_CFG, spans)
    assert not skips
    assert len(masks) == len(decoys) == 2
    for m, d in zip(masks, decoys):
        assert (m[2] - m[0], m[3] - m[1]) == (d[2] - d[0], d[3] - d[1])     # приманка того же размера
        assert all(inter(d, b) == 0 for b in boxes)                         # и не на целях
        assert 100 <= d[1] and d[3] <= 180                                  # по высоте — в пределах целей
    assert any(inter(m, b) == area(m) for m in masks for b in boxes)        # маски — внутри рамок
    assert not np.array_equal(out, img)
