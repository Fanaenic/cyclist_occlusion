"""Окклюзионная аугментация при обучении Ultralytics: маски режимов (configs/masks.yaml, training) на рамках целей.

Маска накладывается в OcclusionDataset.update_labels_info — на загруженное (масштабированное) изображение
до mosaic и прочих аугментаций Ultralytics; mosaic берёт свои плитки через тот же метод, поэтому маски
получают и они. Каждая цель маскируется независимо с вероятностью p режима; у режимов с приманками на каждую
маску на фон ставится такая же заплатка (masks.apply_training_masks).
Запрещённые области — все размеченные люди, rider, bike, cyclist и ignore-области кадра из
datasets_yolo/<ds>/forbidden/ (готовит src/to_yolo.py), а не только отобранные цели. Если чистого места для
приманки или источника фоновой заливки нет, воздействие пропускается; пропуски пишутся в skip_log.

Воспроизводимость при workers > 0: маска зависит только от (seed, номер изображения, эпоха). Эпоху сообщает
EpochSampler — он живёт в главном процессе и кодирует её в индекс (эпоха · N + номер), поэтому ни порядок
работы загрузчиков, ни их упреждающая выборка следующих батчей на маски не влияют.
"""
import hashlib
import os
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

from masks import apply_training_masks


class EpochSampler(Sampler):
    """Оборачивает сэмплер Ultralytics: каждый проход — новая эпоха, индекс = эпоха · n + номер изображения."""

    def __init__(self, base, n):
        self.base, self.n, self.epoch = base, n, -1

    def __iter__(self):
        self.epoch += 1
        for i in self.base:
            yield self.epoch * self.n + i

    def __len__(self):
        return len(self.base)


def forbidden_path(im_file):
    """datasets_yolo/<ds>/images/<part>/<файл> -> datasets_yolo/<ds>/forbidden/<part>/<имя>.txt (как labels в Ultralytics)."""
    p = Path(im_file)
    return p.parent.parent.parent / "forbidden" / p.parent.name / f"{p.stem}.txt"


def read_forbidden(im_file):
    """Рамки всех размеченных людей/велосипедов и ignore-областей кадра в пикселях исходного изображения.

    Файл готовит src/to_yolo.py; его отсутствие — ошибка (иначе приманки могли бы встать на неразмеченный объект).
    """
    path = forbidden_path(im_file)
    if not path.exists():
        raise FileNotFoundError(f"Нет файла запрещённых областей {path} — пересоберите набор src/to_yolo.py")
    return [tuple(map(float, ln.split()[:4])) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def to_pixel_box(b, w, h):
    """Пиксельная рамка, обрезанная по кадру w x h (вход — x1, y1, x2, y2 в пикселях)."""
    x0 = min(max(int(b[0]), 0), w - 1)
    y0 = min(max(int(b[1]), 0), h - 1)
    return x0, y0, min(max(x0 + 1, int(np.ceil(b[2]))), w), min(max(y0 + 1, int(np.ceil(b[3]))), h)


class OcclusionDataset(YOLODataset):
    # {"regime", "cfg": весь configs/masks.yaml, "seed", "span_pool",
    #  "mask_log": каталог | None — журнал масок (проверка воспроизводимости),
    #  "skip_log": каталог | None — журнал пропущенных воздействий (эпоха, изображение, вид, причина)}
    occ = None

    def __getitem__(self, k):
        self._occ_epoch, index = divmod(int(k), len(self.labels))
        return super().__getitem__(index)

    def get_image_and_label(self, index, full=False):
        self._occ_index = index
        return super().get_image_and_label(index, full)

    def _forbidden(self, index):
        cache = self.__dict__.setdefault("_occ_forbidden", {})
        if index not in cache:
            cache[index] = read_forbidden(self.im_files[index])
        return cache[index]

    def update_labels_info(self, label):
        o = self.occ
        boxes = label.get("bboxes")
        if o and boxes is not None and len(boxes):
            epoch = getattr(self, "_occ_epoch", 0)
            index = self._occ_index
            rng = np.random.default_rng([o["seed"], index, epoch])
            h, w = label["resized_shape"]
            h0, w0 = label["ori_shape"]
            sx, sy = w / w0, h / h0
            px = [to_pixel_box(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h), w, h)
                  for cx, cy, bw, bh in boxes]
            fb = [to_pixel_box((b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy), w, h) for b in self._forbidden(index)]
            label["img"], masks, decoys, skips = apply_training_masks(label["img"], px, rng, o["regime"], o["cfg"],
                                                                      o.get("span_pool"), fb)
            name = Path(self.im_files[index]).name
            if o.get("mask_log"):
                # журнал для проверки воспроизводимости: эпоха, изображение, маски, приманки, хеш пикселей
                h_img = hashlib.sha1(np.ascontiguousarray(label["img"]).tobytes()).hexdigest()[:16]
                with open(Path(o["mask_log"]) / f"{os.getpid()}.txt", "a", encoding="utf-8") as fh:
                    fh.write(f"{epoch}\t{name}\t{masks}\t{decoys}\t{h_img}\n")
            if skips and o.get("skip_log"):
                with open(Path(o["skip_log"]) / f"{os.getpid()}.txt", "a", encoding="utf-8") as fh:
                    fh.writelines(f"{epoch}\t{name}\t{s['kind']}\t{s['reason']}\n" for s in skips)
        return super().update_labels_info(label)


class OcclusionTrainer(DetectionTrainer):
    occ = None            # задаётся перед model.train(trainer=OcclusionTrainer)

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        if mode == "train" and self.occ and self.occ["cfg"]["training"][self.occ["regime"]].get("p", 0) > 0:
            ds.__class__ = OcclusionDataset
            ds.occ = self.occ
        return ds

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
        if isinstance(loader.dataset, OcclusionDataset):
            batch_sampler = loader.batch_sampler.sampler          # _RepeatSampler -> BatchSampler
            batch_sampler.sampler = EpochSampler(batch_sampler.sampler, len(loader.dataset.labels))
        return loader
