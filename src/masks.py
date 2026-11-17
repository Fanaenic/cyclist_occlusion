"""Генератор прямоугольных масок окклюзии внутри рамки объекта.

Библиотечная часть (используется при обучении и оценке):
  pixel_box, make_mask, apply_mask         — маска на заданную долю площади рамки
  sample_training_mask, random_erasing_ore — режимы обучения proposed и ORE
  apply_training_masks, place_patch        — маски обучения и приманки в типичном для целей положении

Командная часть:
  generate  — один раз построить маски и контрольные заплатки для calib/test -> splits/{masks,control}_{ds}_{part}.csv
  check     — программно проверить фактическую долю перекрытия и соседей -> reports/masks_check.*
  examples  — картинки каждого уровня 0/30/50/70/90 % и режимов обучения -> reports/examples/
  placement — гистограммы положения и размера: цели / маски / приманки / контроль -> reports/placement.*

Запуск: python src/masks.py {generate,check,examples,placement} [--config configs/masks.yaml] [--dataset kitti|bdd]
        [--limit N]
"""
import argparse
import csv
import json
import math
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

MASK_FIELDS = ["dataset", "part", "image", "object_id", "level",
               "bx0", "by0", "bx1", "by1", "mx0", "my0", "mx1", "my1",
               "target_frac", "actual_frac", "mask_seed", "neighbor_max_cover", "neighbor_flag",
               "sx", "sy", "src_clean"]
# Контрольный набор: заплатки только на фоне (вне всех рамок) — проверка, не учится ли модель на заплатку.
CONTROL_FIELDS = ["dataset", "part", "image", "object_id", "dx0", "dy0", "dx1", "dy1", "sx", "sy",
                  "mask_seed", "clean"]


# ---------------------------------------------------------------- геометрия

def pixel_box(x1, y1, x2, y2):
    """Рамка в целых пикселях: [floor(x1), ceil(x2)) x [floor(y1), ceil(y2))."""
    x0, y0 = int(math.floor(x1)), int(math.floor(y1))
    return max(0, x0), max(0, y0), max(x0 + 1, int(math.ceil(x2))), max(y0 + 1, int(math.ceil(y2)))


def area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def inter(a, b):
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def rng_for(seed, *key):
    """Генератор, зависящий только от seed и ключа объекта (не от порядка обхода)."""
    return np.random.default_rng([seed, zlib.crc32("|".join(map(str, key)).encode())])


def mask_size(W, H, frac, aspect):
    """Целые (w, h) маски с w*h как можно ближе к frac*W*H, w<=W, h<=H, форма близка к aspect=w/h."""
    target = frac * W * H
    h_f = math.sqrt(target / aspect)
    best = None
    for h in range(1, H + 1):
        w = min(W, max(1, round(target / h)))
        key = (abs(w * h - target), abs(math.log((w / h) / aspect)), abs(h - h_f))
        if best is None or key < best[0]:
            best = (key, w, h)
    return best[1], best[2]


def make_mask(box, frac, rng, shape_cfg, neighbors=(), nb_cfg=None):
    """Маска (x0, y0, x1, y1) внутри box на долю frac его площади.

    neighbors — пиксельные рамки соседей; при policy=avoid перебираются форма и положение,
    чтобы под маской было не больше max_cover площади каждого соседа.
    Возвращает (маска, наибольшая доля соседа под маской, флаг «соседа избежать не удалось»).
    """
    bx0, by0, bx1, by1 = box
    W, H = bx1 - bx0, by1 - by0
    if frac <= 0:
        return (bx0, by0, bx0, by0), 0.0, False
    lo, hi = shape_cfg["aspect_ratio"]
    avoid = nb_cfg is not None and nb_cfg["policy"] == "avoid" and len(neighbors) > 0
    tries = nb_cfg["max_tries"] if avoid else 1
    best = None
    for _ in range(tries):
        aspect = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        w, h = mask_size(W, H, frac, aspect)
        if shape_cfg["position"] == "center":
            x0, y0 = bx0 + (W - w) // 2, by0 + (H - h) // 2
        else:
            x0 = bx0 + int(rng.integers(0, W - w + 1))
            y0 = by0 + int(rng.integers(0, H - h + 1))
        m = (x0, y0, x0 + w, y0 + h)
        cover = max((inter(m, n) / area(n) for n in neighbors if area(n)), default=0.0)
        if best is None or cover < best[1]:
            best = (m, cover)
        if not avoid or cover <= nb_cfg["max_cover"]:
            break
    m, cover = best
    flag = avoid and cover > nb_cfg["max_cover"]
    return m, cover, flag


# ---------------------------------------------------------------- заливка

def fill_patch(shape, fill_cfg, rng):
    if fill_cfg["mode"] == "constant":
        return np.broadcast_to(np.array(fill_cfg["constant"], dtype=np.uint8), shape).copy()
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def pick_background(W, H, w, h, avoid, rng, tries):
    """Левый верхний угол участка w x h внутри кадра W x H, не пересекающего рамки avoid.

    Возвращает (sx, sy, clean): clean = False, если чистого участка не нашлось (берётся с наименьшим пересечением).
    """
    w, h = min(w, W), min(h, H)
    best = None
    for _ in range(tries):
        sx, sy = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
        ov = sum(inter((sx, sy, sx + w, sy + h), a) for a in avoid)
        if best is None or ov < best[2]:
            best = (sx, sy, ov)
        if ov == 0:
            break
    return best[0], best[1], best[2] == 0


def pick_like_targets(W, H, w, h, avoid, rng, tries, span_pool):
    """Положение заплатки w x h, как у масок на реальных целях: берётся вертикальный диапазон [y1, y2] случайной
    цели train (span_pool — пары в долях высоты кадра), заплатка ставится в него по вертикали равномерно
    (если выше диапазона — по нижнему краю), x — равномерно по кадру; вне рамок avoid.

    Возвращает (x0, y0, clean): clean = False, если чистого места не нашлось (берётся с наименьшим пересечением).
    """
    w, h = min(w, W), min(h, H)
    pool = np.asarray(span_pool, dtype=float)
    fits = pool[(pool[:, 1] - pool[:, 0]) * H >= h]       # диапазоны целей, в которые заплатка помещается
    pool = fits if len(fits) else pool
    best = None
    for _ in range(tries):
        y1, y2 = pool[int(rng.integers(0, len(pool)))]
        t0, t1 = int(round(y1 * H)), int(round(y2 * H))
        y0 = int(rng.integers(t0, t1 - h + 1)) if t1 - t0 >= h else t1 - h
        y0 = min(max(y0, 0), H - h)
        x0 = int(rng.integers(0, W - w + 1))
        ov = sum(inter((x0, y0, x0 + w, y0 + h), a) for a in avoid)
        if best is None or ov < best[2]:
            best = (x0, y0, ov)
        if ov == 0:
            break
    return best[0], best[1], best[2] == 0


def place_patch(W, H, w, h, avoid, rng, cfg, span_pool):
    """Положение приманки / контрольной заплатки по правилу placement.mode из configs/masks.yaml."""
    tries = cfg["background"]["max_tries"]
    if cfg["placement"]["mode"] == "like_targets":
        return pick_like_targets(W, H, w, h, avoid, rng, tries, span_pool)
    return pick_background(W, H, w, h, avoid, rng, tries)


def apply_mask(img, m, fill_cfg, mask_seed, src=None, source_img=None):
    """Закрыть прямоугольник m на копии изображения.

    fill.mode: noise / constant — заливка (шум детерминирован mask_seed);
    background — копия участка фона того же изображения с левым верхним углом src
    (берётся из source_img — исходного кадра без масок; по умолчанию из img).
    """
    x0, y0, x1, y1 = (int(v) for v in m)
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    out = img.copy()
    if x1 > x0 and y1 > y0:
        if fill_cfg["mode"] == "background":
            base = img if source_img is None else source_img
            sx, sy = (int(v) for v in src)
            out[y0:y1, x0:x1] = base[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
        else:
            out[y0:y1, x0:x1] = fill_patch((y1 - y0, x1 - x0) + img.shape[2:], fill_cfg,
                                           np.random.default_rng(int(mask_seed)))
    return out


# ---------------------------------------------------------------- режимы обучения

def sample_training_mask(box, rng, regime, shape_cfg, neighbors=(), nb_cfg=None):
    """proposed: с вероятностью p выбрать уровень из regime['levels'] (по весам) и построить маску."""
    if regime.get("p", 0) <= 0 or rng.random() >= regime["p"]:
        return None
    w = np.asarray(regime["weights"], dtype=float)
    frac = float(rng.choice(regime["levels"], p=w / w.sum()))
    m, _, _ = make_mask(box, frac, rng, shape_cfg, neighbors, nb_cfg)
    return m


def random_erasing_ore(box, rng, ore):
    """Object-aware Random Erasing (Zhong et al.): площадь Se~U(area), форма re~U(aspect), повтор до попадания в рамку."""
    if rng.random() >= ore["p"]:
        return None
    bx0, by0, bx1, by1 = box
    W, H = bx1 - bx0, by1 - by0
    for _ in range(ore["max_tries"]):
        se = rng.uniform(*ore["area"]) * W * H
        re = rng.uniform(*ore["aspect_ratio"])
        h, w = int(round(math.sqrt(se * re))), int(round(math.sqrt(se / re)))
        if 0 < w < W and 0 < h < H:
            x0 = bx0 + int(rng.integers(0, W - w))
            y0 = by0 + int(rng.integers(0, H - h))
            return (x0, y0, x0 + w, y0 + h)
    return None


def training_geometry(boxes, W, H, rng, regime, cfg, span_pool=None):
    """Маски режима обучения на целях и приманки (без изображения). Возвращает (маски, приманки)."""
    rcfg = cfg["training"][regime]
    masks, decoys = [], []
    for i, b in enumerate(boxes):
        if regime == "ore":
            m = random_erasing_ore(b, rng, rcfg)
        else:
            nbs = [x for j, x in enumerate(boxes) if j != i]
            m = sample_training_mask(b, rng, rcfg, cfg["shape"], nbs, cfg["neighbors"])
        if m:
            masks.append(m)
    if rcfg.get("decoys", True):
        for m in masks:
            w, h = m[2] - m[0], m[3] - m[1]
            dx, dy, _ = place_patch(W, H, w, h, list(boxes) + decoys, rng, cfg, span_pool)
            decoys.append((dx, dy, dx + w, dy + h))
    return masks, decoys


def apply_training_masks(img, boxes, rng, regime, cfg, span_pool=None):
    """Маски режима обучения на всех целях кадра + столько же заплаток-приманок на фоне.

    boxes — пиксельные рамки целей; regime — "proposed", "ore" или "proposed_noise"; cfg — весь configs/masks.yaml;
    span_pool — диапазоны [y1, y2] целей train (доли высоты кадра) для размещения приманок (placement.mode: like_targets).
    Приманка — участок того же размера, что маска, вне рамок целей, в типичном для целей положении,
    закрытый так же (фоном из другого места). Так присутствие заплатки не указывает на класс объекта.
    Заливка — training.<режим>.fill, если задана (proposed_noise: шум), иначе общая fill.
    Возвращает (изображение, список масок, список приманок).
    """
    H, W = img.shape[:2]
    tries = cfg["background"]["max_tries"]
    fill = cfg["training"][regime].get("fill", cfg["fill"])
    out = img.copy()
    masks, decoys = training_geometry(boxes, W, H, rng, regime, cfg, span_pool)
    for m in masks + decoys:
        w, h = m[2] - m[0], m[3] - m[1]
        src = pick_background(W, H, w, h, list(boxes) + [m], rng, tries)[:2]
        out = apply_mask(out, m, fill, int(rng.integers(0, 2**31 - 1)), src, img)
    return out, masks, decoys


# ---------------------------------------------------------------- загрузка объектов

def load_split(cfg, ds, part):
    return set((Path(cfg["splits_dir"]) / f"{ds}_{part}.txt").read_text(encoding="utf-8").split())


def kitti_objects(cfg, part):
    """Целевые объекты части и рамки соседей по изображениям."""
    d = cfg["datasets"]["kitti"]
    images = load_split(cfg, "kitti", part)
    targets = []
    with open(d["selected_file"], encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["image"] in images:
                targets.append((r["image"], f"{r['image']}:{r['line']}",
                                pixel_box(*(float(r[k]) for k in ("left", "top", "right", "bottom")))))
    cats = set(cfg["neighbors"]["categories"]["kitti"])
    boxes = defaultdict(list)
    for img in {t[0] for t in targets}:
        for i, ln in enumerate((Path(d["label_dir"]) / f"{img}.txt").read_text().splitlines()):
            p = ln.split()
            if p and p[0] in cats:
                boxes[img].append((f"{img}:{i}", pixel_box(*map(float, p[4:8]))))
    return targets, boxes, {}


def bdd_objects(cfg, part):
    d = cfg["datasets"]["bdd"]
    src = d["sources"][part]
    images = load_split(cfg, "bdd", part)
    targets, own_bike = [], {}
    with open(d["selected_file"].format(source=src), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["image"] in images and r["last_step"] == d["final_step"]:
                targets.append((r["image"], r["rider_id"],
                                pixel_box(*(float(r[f"target_{k}"]) for k in ("x1", "y1", "x2", "y2")))))
                own_bike[r["rider_id"]] = r["bike_id"]
    need = {t[0] for t in targets}
    cats = set(cfg["neighbors"]["categories"]["bdd"])
    boxes = defaultdict(list)
    with open(Path(d["labels_dir"]) / d["label_file"].format(source=src), encoding="utf-8") as fh:
        for it in json.load(fh):
            if it["name"] not in need:
                continue
            for lb in it.get("labels") or []:
                if lb.get("category") in cats and lb.get("box2d"):
                    b = lb["box2d"]
                    boxes[it["name"]].append((str(lb["id"]), pixel_box(b["x1"], b["y1"], b["x2"], b["y2"])))
    return targets, boxes, own_bike


LOADERS = {"kitti": kitti_objects, "bdd": bdd_objects}


def image_size(cfg, ds, part, img):
    """(W, H) кадра: BDD100K — фиксированный размер, KITTI — из файла изображения."""
    if ds == "bdd":
        return tuple(cfg["datasets"]["bdd"]["image_size"])
    from PIL import Image
    d = cfg["datasets"]["kitti"]
    with Image.open(Path(d["image_dir"]) / f"{img}{d['image_ext']}") as im:
        return im.size


def target_span_pool(cfg, ds):
    """Вертикальные диапазоны [y1, y2] целевых рамок части pool_part в долях высоты кадра
    (кэш — splits/target_spans_{ds}.json)."""
    path = Path(cfg["placement"]["pool_file"].format(dataset=ds))
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["spans"]
    part = cfg["placement"]["pool_part"]
    targets, _, _ = LOADERS[ds](cfg, part)
    spans = []
    for img, _, box in sorted(targets):
        H = image_size(cfg, ds, part, img)[1]
        spans.append([round(box[1] / H, 5), round(box[3] / H, 5)])
    path.write_text(json.dumps({"dataset": ds, "part": part, "n": len(spans), "spans": spans}), encoding="utf-8")
    return spans


# ---------------------------------------------------------------- команды

def generate(cfg, ds, limit):
    seed = cfg["fixed_masks"]["seed"]
    out = Path(cfg["splits_dir"])
    span_pool = target_span_pool(cfg, ds)
    for part in cfg["fixed_masks"]["parts"]:
        targets, boxes, own_bike = LOADERS[ds](cfg, part)
        targets.sort()
        if limit:
            targets = targets[:limit]
        rows, control = [], []
        tries = cfg["background"]["max_tries"]
        for img, oid, box in targets:
            skip = {oid, own_bike.get(oid)}
            nbs = [b for nid, b in boxes[img] if nid not in skip]
            all_boxes = [b for _, b in boxes[img]] + [box]      # все размеченные люди/велосипеды кадра
            W, H = image_size(cfg, ds, part, img)
            for level in cfg["levels"]:
                rng = rng_for(seed, ds, part, img, oid, level)
                m, cover, flag = make_mask(box, level, rng, cfg["shape"], nbs, cfg["neighbors"])
                mask_seed = int(rng.integers(0, 2**31 - 1))
                sx = sy = clean = ""
                if level > 0:
                    sx, sy, clean = pick_background(W, H, m[2] - m[0], m[3] - m[1], all_boxes, rng, tries)
                    clean = int(clean)
                rows.append({
                    "dataset": ds, "part": part, "image": img, "object_id": oid, "level": level,
                    "bx0": box[0], "by0": box[1], "bx1": box[2], "by1": box[3],
                    "mx0": m[0], "my0": m[1], "mx1": m[2], "my1": m[3],
                    "target_frac": level, "actual_frac": round(area(m) / area(box), 5),
                    "mask_seed": mask_seed, "neighbor_max_cover": round(cover, 4), "neighbor_flag": int(flag),
                    "sx": sx, "sy": sy, "src_clean": clean,
                })
                if level == cfg["control"]["level"]:
                    # заплатка того же размера на фоне (вне всех рамок, положение — как у целей) и источник заливки
                    w_, h_ = m[2] - m[0], m[3] - m[1]
                    dx, dy, ok1 = place_patch(W, H, w_, h_, all_boxes, rng, cfg, span_pool)
                    d = (dx, dy, dx + w_, dy + h_)
                    cx, cy, ok2 = pick_background(W, H, w_, h_, all_boxes + [d], rng, tries)
                    control.append({"dataset": ds, "part": part, "image": img, "object_id": oid,
                                    "dx0": d[0], "dy0": d[1], "dx1": d[2], "dy1": d[3], "sx": cx, "sy": cy,
                                    "mask_seed": int(rng.integers(0, 2**31 - 1)), "clean": int(ok1 and ok2)})
        path = out / f"masks_{ds}_{part}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MASK_FIELDS)
            w.writeheader()
            w.writerows(rows)
        with open(out / f"control_{ds}_{part}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CONTROL_FIELDS)
            w.writeheader()
            w.writerows(control)
        print(f"{ds} {part}: {len(targets)} объектов x {len(cfg['levels'])} уровней -> {path}; "
              f"контрольных заплаток: {len(control)}")


def check_control(cfg, datasets):
    """Контрольные заплатки: внутри кадра, вне всех рамок, по вертикали — в пределах [min y1, max y2] целей train."""
    rows = []
    for ds in datasets:
        pool = target_span_pool(cfg, ds)
        lo, hi = min(s[0] for s in pool), max(s[1] for s in pool)
        for part in cfg["fixed_masks"]["parts"]:
            path = Path(cfg["splits_dir"]) / f"control_{ds}_{part}.csv"
            if not path.exists():
                continue
            _, boxes, _ = LOADERS[ds](cfg, part)
            n = outside = on_object = y2_out = 0
            with open(path, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    d = tuple(int(r[k]) for k in ("dx0", "dy0", "dx1", "dy1"))
                    W, H = image_size(cfg, ds, part, r["image"])
                    n += 1
                    outside += not (d[0] >= 0 and d[1] >= 0 and d[2] <= W and d[3] <= H)
                    on_object += any(inter(d, b) for _, b in boxes[r["image"]])
                    y2_out += not (d[1] / H >= lo - 1 / H and d[3] / H <= hi + 1 / H)
            rows.append({"dataset": ds, "part": part, "patches": n, "outside_frame": outside,
                         "on_object": on_object, "y2_outside_target_range": y2_out})
    return rows


def geometry_samples(cfg, ds):
    """Положение и размер (y-центр / H, площадь / (W·H)): цели, маски, приманки (обучение) и контроль (test)."""
    out = defaultdict(list)

    def add(kind, b, W, H):
        out[kind].append(((b[1] + b[3]) / 2 / H, area(b) / (W * H)))

    span_pool = target_span_pool(cfg, ds)
    tr = cfg["training"]
    targets, _, _ = LOADERS[ds](cfg, "train")
    by_img = defaultdict(list)
    for img, _, box in targets:
        by_img[img].append(box)
    for img, bxs in sorted(by_img.items()):
        W, H = image_size(cfg, ds, "train", img)
        rng = rng_for(tr["seed"], "placement", ds, img)
        masks, decoys = training_geometry(bxs, W, H, rng, "proposed", cfg, span_pool)
        for b in bxs:
            add("train: цели", b, W, H)
        for m in masks:
            add("train: маски proposed", m, W, H)
        for d in decoys:
            add("train: приманки", d, W, H)
    part = "test"
    with open(Path(cfg["splits_dir"]) / f"masks_{ds}_{part}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            W, H = image_size(cfg, ds, part, r["image"])
            if float(r["level"]) == 0:
                add("test: цели", tuple(int(r[k]) for k in ("bx0", "by0", "bx1", "by1")), W, H)
            else:
                add("test: маски", tuple(int(r[k]) for k in ("mx0", "my0", "mx1", "my1")), W, H)
    with open(Path(cfg["splits_dir"]) / f"control_{ds}_{part}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            W, H = image_size(cfg, ds, part, r["image"])
            add("test: контрольные заплатки", tuple(int(r[k]) for k in ("dx0", "dy0", "dx1", "dy1")), W, H)
    return out


def ks(a, b):
    """Статистика Колмогорова–Смирнова (максимальная разность эмпирических функций распределения)."""
    a, b = np.sort(a), np.sort(b)
    grid = np.r_[a, b]
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def placement_report(cfg, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = Path(cfg["report_dir"])
    lines = ["# Положение и размер заплаток: цели / маски / приманки / контроль", "",
             f"Правило размещения приманок и контрольных заплаток: `{cfg['placement']['mode']}` "
             f"(вертикальный диапазон случайной цели {cfg['placement']['pool_part']}, заплатка в нём — равномерно, "
             "x — равномерно). Приманки обучения — моделирование режима proposed на целях train (без изображений). "
             "KS — статистика Колмогорова–Смирнова по y-центру: относительно целей и относительно масок той же части "
             "(приманки и контроль должны быть похожи на маски: те же размеры, то же положение по высоте).", "",
             "| датасет | набор | n | y-центр: медиана | площадь: медиана, % кадра | KS к целям | KS к маскам |",
             "|---|---|---|---|---|---|---|"]
    for ds in datasets:
        g = geometry_samples(cfg, ds)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for kind, vals in g.items():
            y, a = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
            tr = kind.startswith("train")
            k_t = ks(y, np.array([v[0] for v in g["train: цели" if tr else "test: цели"]]))
            k_m = ks(y, np.array([v[0] for v in g["train: маски proposed" if tr else "test: маски"]]))
            lines.append(f"| {ds} | {kind} | {len(vals)} | {np.median(y):.3f} | {100 * np.median(a):.3f} | "
                         f"{k_t:.3f} | {k_m:.3f} |")
            style = "-" if "цели" in kind else "--" if "train" in kind else ":"
            axes[0].hist(y, bins=np.linspace(0, 1, 41), density=True, histtype="step", linestyle=style,
                         linewidth=1.6, label=kind)
            axes[1].hist(np.log10(a), bins=np.linspace(-5, -0.5, 41), density=True, histtype="step",
                         linestyle=style, linewidth=1.6)
        axes[0].set_xlabel("y-центр / высота кадра (0 — верх)")
        axes[1].set_xlabel("log10(площадь / площадь кадра)")
        axes[0].set_ylabel("плотность")
        axes[0].legend(fontsize=7)
        fig.suptitle(ds)
        fig.tight_layout()
        fig.savefig(rep / f"placement_{ds}.png", dpi=110)
        plt.close(fig)
        lines.append("")
        lines.append(f"![](placement_{ds}.png)")
        lines.append("")
    crow = check_control(cfg, datasets)
    lines += ["## Проверка контрольных заплаток", "",
              "| датасет | часть | заплаток | за кадром | на размеченном объекте | вне [min y1, max y2] целей train |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {r['dataset']} | {r['part']} | {r['patches']} | {r['outside_frame']} | {r['on_object']} | "
              f"{r['y2_outside_target_range']} |" for r in crow]
    (rep / "placement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return sum(r["outside_frame"] + r["on_object"] + r["y2_outside_target_range"] for r in crow)


def check(cfg, datasets):
    """Независимая проверка по сохранённым файлам: доля, попадание в рамку, соседи."""
    tol = cfg["tolerance_pp"] / 100
    summary, fails = [], []
    for ds in datasets:
        for part in cfg["fixed_masks"]["parts"]:
            path = Path(cfg["splits_dir"]) / f"masks_{ds}_{part}.csv"
            if not path.exists():
                continue
            _, boxes, own_bike = LOADERS[ds](cfg, part)
            by_level = defaultdict(list)
            with open(path, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    b = tuple(int(r[k]) for k in ("bx0", "by0", "bx1", "by1"))
                    m = tuple(int(r[k]) for k in ("mx0", "my0", "mx1", "my1"))
                    frac = inter(m, b) / area(b)
                    inside = area(m) == inter(m, b)
                    skip = {r["object_id"], own_bike.get(r["object_id"])}
                    nbs = [nb for nid, nb in boxes[r["image"]] if nid not in skip]
                    cover = max((inter(m, n) / area(n) for n in nbs if area(n)), default=0.0)
                    ok = abs(frac - float(r["target_frac"])) <= tol and inside
                    # источник заливки (фон) не должен задевать ни одну размеченную рамку кадра
                    dirty = False
                    if cfg["fill"]["mode"] == "background" and float(r["level"]) > 0:
                        sx, sy = int(r["sx"]), int(r["sy"])
                        s = (sx, sy, sx + m[2] - m[0], sy + m[3] - m[1])
                        dirty = any(inter(s, a) for a in [nb for _, nb in boxes[r["image"]]] + [b])
                    by_level[float(r["level"])].append((frac, ok, cover > cfg["neighbors"]["max_cover"], len(nbs) > 0,
                                                        dirty))
                    if not ok:
                        fails.append({**r, "checked_frac": round(frac, 5), "inside": inside})
            for level, vals in sorted(by_level.items()):
                fr = [v[0] for v in vals]
                summary.append({
                    "dataset": ds, "part": part, "level": level, "objects": len(vals),
                    "mean_frac": round(float(np.mean(fr)), 4), "min_frac": round(min(fr), 4), "max_frac": round(max(fr), 4),
                    "max_abs_err_pp": round(max(abs(f - level) for f in fr) * 100, 2),
                    "fails": sum(not v[1] for v in vals),
                    "with_neighbors": sum(v[3] for v in vals),
                    "neighbor_over_limit": sum(v[2] for v in vals),
                    "source_on_object": sum(v[4] for v in vals),
                })
    rep = Path(cfg["report_dir"])
    with open(rep / "masks_check.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    if fails:
        with open(rep / "masks_check_fails.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fails[0].keys()))
            w.writeheader()
            w.writerows(fails)
    lines = ["# Проверка тестовых масок", "",
             f"Допуск: ±{cfg['tolerance_pp']} п.п.; маска должна целиком лежать в рамке. "
             f"Соседи: доля площади соседней рамки под маской ≤ {cfg['neighbors']['max_cover']} "
             f"(политика `{cfg['neighbors']['policy']}`).", "",
             f"Заливка: `{cfg['fill']['mode']}`; для `background` проверяется, что участок-источник не задевает "
             "ни одну размеченную рамку человека/велосипеда кадра.", "",
             "| датасет | часть | уровень | объектов | средняя доля | мин | макс | макс. ошибка, п.п. | вне допуска | с соседями | сосед закрыт сверх порога | источник на объекте |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {s['dataset']} | {s['part']} | {s['level']:.0%} | {s['objects']} | {s['mean_frac']:.4f} | {s['min_frac']:.4f} | "
              f"{s['max_frac']:.4f} | {s['max_abs_err_pp']} | {s['fails']} | {s['with_neighbors']} | {s['neighbor_over_limit']} | "
              f"{s['source_on_object']} |"
              for s in summary]
    lines += ["", f"Итого вне допуска: {len(fails)}" + (" (см. masks_check_fails.csv)" if fails else "") + "."]
    (rep / "masks_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for s in summary:
        print(f"{s['dataset']:5s} {s['part']:5s} {s['level']:.0%} n={s['objects']:4d} mean={s['mean_frac']:.4f} "
              f"err<={s['max_abs_err_pp']}pp fails={s['fails']} nb_over={s['neighbor_over_limit']}")
    print(f"Вне допуска: {len(fails)}")
    return len(fails)


def examples(cfg, ds):
    """Одни и те же объекты на всех тестовых уровнях + режимы обучения proposed и ORE."""
    import cv2

    d = cfg["datasets"][ds]
    part = "test"
    path = Path(cfg["splits_dir"]) / f"masks_{ds}_{part}.csv"
    rows = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[(r["image"], r["object_id"])][float(r["level"])] = r
    src = d.get("sources", {}).get(part, "")
    img_dir = Path(d["image_dir"].format(source=src))
    keys = [k for k in sorted(rows) if (img_dir / f"{k[0]}{d['image_ext']}").exists()]
    if not keys:
        print(f"{ds}: нет изображений в {img_dir} — примеры не построены")
        return
    out = Path(cfg["examples_dir"])
    out.mkdir(parents=True, exist_ok=True)

    def height(k):
        r = rows[k][0.0]
        return int(r["by1"]) - int(r["by0"])

    # три объекта: маленький, средний, крупный (квантили высоты)
    by_h = sorted(keys, key=height)
    picks = {"small": by_h[len(by_h) // 10], "medium": by_h[len(by_h) // 2], "large": by_h[-max(1, len(by_h) // 20)]}
    # и объект, у которого маске пришлось обходить соседа
    # объект, чья рамка сильнее всего пересекается с соседней: на нём видно, как маска обходит соседа
    _, boxes, own_bike = LOADERS[ds](cfg, part)

    def nb_overlap(k):
        b = tuple(int(rows[k][0.0][c]) for c in ("bx0", "by0", "bx1", "by1"))
        skip = {k[1], own_bike.get(k[1])}
        return max((inter(b, n) / area(n) for nid, n in boxes[k[0]] if nid not in skip and area(n)), default=0.0)

    best_nb = max(keys, key=nb_overlap)
    if nb_overlap(best_nb) > 0:
        picks["neighbor"] = best_nb

    tile_h = 220
    for name, k in picks.items():
        img = cv2.imread(str(img_dir / f"{k[0]}{d['image_ext']}"))
        tiles = []
        for level in cfg["levels"]:
            r = rows[k][level]
            b = tuple(int(r[c]) for c in ("bx0", "by0", "bx1", "by1"))
            m = tuple(int(r[c]) for c in ("mx0", "my0", "mx1", "my1"))
            im = apply_mask(img, m, cfg["fill"], r["mask_seed"], (r["sx"], r["sy"]))
            pad = max(b[2] - b[0], b[3] - b[1])
            x0, y0 = max(0, b[0] - pad), max(0, b[1] - pad)
            x1, y1 = min(im.shape[1], b[2] + pad), min(im.shape[0], b[3] + pad)
            cv2.rectangle(im, b[:2], (b[2] - 1, b[3] - 1), (0, 255, 0), 1)
            crop = im[y0:y1, x0:x1]
            s = tile_h / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), tile_h), interpolation=cv2.INTER_NEAREST)
            label = f"{level:.0%} -> {float(r['actual_frac']):.1%}"
            cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(crop, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            tiles.append(crop)
        cv2.imwrite(str(out / f"{ds}_levels_{name}.png"), np.hstack(tiles))

        if name == "neighbor":
            # полный кадр: рамка цели (зелёная), соседи (красные), маска 90 %
            r = rows[k][max(cfg["levels"])]
            full = apply_mask(img, tuple(int(r[c]) for c in ("mx0", "my0", "mx1", "my1")), cfg["fill"], r["mask_seed"],
                              (r["sx"], r["sy"]))
            for nid, nbox in boxes[k[0]]:
                if nid not in {k[1], own_bike.get(k[1])}:
                    cv2.rectangle(full, nbox[:2], (nbox[2] - 1, nbox[3] - 1), (0, 0, 255), 1)
            b = tuple(int(r[c]) for c in ("bx0", "by0", "bx1", "by1"))
            cv2.rectangle(full, b[:2], (b[2] - 1, b[3] - 1), (0, 255, 0), 1)
            cv2.imwrite(str(out / f"{ds}_neighbor_full.png"), full)

    # режимы обучения: полный кадр, маски на целях (зелёные рамки) и приманки на фоне (красные), p = 1
    k = picks["large"]
    img = cv2.imread(str(img_dir / f"{k[0]}{d['image_ext']}"))
    tboxes = [tuple(int(r[c]) for c in ("bx0", "by0", "bx1", "by1"))
              for (im_, _), lv in rows.items() if im_ == k[0] for r in [lv[0.0]]]
    tr = cfg["training"]
    for regime in ("proposed", "ore"):
        frames = []
        for i in range(2):
            rng = rng_for(tr["seed"], "example", regime, i)
            ex_cfg = {**cfg, "training": {**tr, regime: {**tr[regime], "p": 1.0}}}
            im, masks_, decoys = apply_training_masks(img, tboxes, rng, regime, ex_cfg, target_span_pool(cfg, ds))
            for m in masks_:
                cv2.rectangle(im, m[:2], (m[2] - 1, m[3] - 1), (0, 255, 0), 2)
            for m in decoys:
                cv2.rectangle(im, m[:2], (m[2] - 1, m[3] - 1), (0, 0, 255), 2)
            frames.append(im)
        cv2.imwrite(str(out / f"{ds}_train_{regime}.png"), np.vstack(frames))

    # контрольный набор: заплатки только на фоне (уровень control.level)
    cpath = Path(cfg["splits_dir"]) / f"control_{ds}_{part}.csv"
    if cpath.exists():
        with open(cpath, encoding="utf-8") as fh:
            crow = [r for r in csv.DictReader(fh) if r["image"] == k[0]]
        im = img.copy()
        for r in crow:
            dbox = tuple(int(r[c]) for c in ("dx0", "dy0", "dx1", "dy1"))
            im = apply_mask(im, dbox, cfg["fill"], r["mask_seed"], (r["sx"], r["sy"]), img)
            cv2.rectangle(im, dbox[:2], (dbox[2] - 1, dbox[3] - 1), (0, 0, 255), 2)
        for b in tboxes:
            cv2.rectangle(im, b[:2], (b[2] - 1, b[3] - 1), (0, 255, 0), 1)
        cv2.imwrite(str(out / f"{ds}_control.png"), im)
    print(f"{ds}: примеры -> {out}/{ds}_*.png ({', '.join(picks)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "check", "examples", "placement"])
    ap.add_argument("--config", default="configs/masks.yaml")
    ap.add_argument("--dataset", choices=["kitti", "bdd"], action="append",
                    help="можно указать несколько раз; по умолчанию оба")
    ap.add_argument("--limit", type=int, default=0, help="только первые N объектов каждой части (smoke test)")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    datasets = args.dataset or ["kitti", "bdd"]
    if args.command == "generate":
        for ds in datasets:
            generate(cfg, ds, args.limit)
    elif args.command == "check":
        raise SystemExit(1 if check(cfg, datasets) else 0)
    elif args.command == "placement":
        raise SystemExit(1 if placement_report(cfg, datasets) else 0)
    else:
        for ds in datasets:
            examples(cfg, ds)


if __name__ == "__main__":
    main()
