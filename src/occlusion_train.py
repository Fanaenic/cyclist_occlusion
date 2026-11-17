"""Окклюзионная аугментация при обучении Ultralytics: маски proposed / ORE на рамках целей.

Маска накладывается в OcclusionDataset.update_labels_info — на загруженное (масштабированное) изображение
до mosaic и прочих аугментаций Ultralytics; mosaic берёт свои плитки через тот же метод, поэтому маски
получают и они. Каждая цель маскируется независимо с вероятностью p режима; на каждую маску на фон
ставится такая же заплатка-приманка (masks.apply_training_masks, configs/masks.yaml).
Соседи и запретные для приманок области при обучении — только цели кадра (прочие классы в YOLO-разметке
не хранятся).

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


class OcclusionDataset(YOLODataset):
    occ = None            # {"regime", "cfg": весь configs/masks.yaml, "seed", "span_pool", "mask_log": каталог | None}

    def __getitem__(self, k):
        self._occ_epoch, index = divmod(int(k), len(self.labels))
        return super().__getitem__(index)

    def get_image_and_label(self, index, full=False):
        self._occ_index = index
        return super().get_image_and_label(index, full)

    def update_labels_info(self, label):
        o = self.occ
        boxes = label.get("bboxes")
        if o and boxes is not None and len(boxes):
            epoch = getattr(self, "_occ_epoch", 0)
            rng = np.random.default_rng([o["seed"], self._occ_index, epoch])
            h, w = label["resized_shape"]
            px = []
            for cx, cy, bw, bh in boxes:                # пиксельная рамка, обрезанная по кадру
                x0 = min(max(int((cx - bw / 2) * w), 0), w - 1)
                y0 = min(max(int((cy - bh / 2) * h), 0), h - 1)
                x1 = min(max(x0 + 1, int(np.ceil((cx + bw / 2) * w))), w)
                y1 = min(max(y0 + 1, int(np.ceil((cy + bh / 2) * h))), h)
                px.append((x0, y0, x1, y1))
            label["img"], masks, decoys = apply_training_masks(label["img"], px, rng, o["regime"], o["cfg"],
                                                               o.get("span_pool"))
            if o.get("mask_log"):
                # журнал для проверки воспроизводимости: эпоха, изображение, маски, приманки, хеш пикселей
                h_img = hashlib.sha1(np.ascontiguousarray(label["img"]).tobytes()).hexdigest()[:16]
                with open(Path(o["mask_log"]) / f"{os.getpid()}.txt", "a", encoding="utf-8") as fh:
                    fh.write(f"{epoch}\t{self.im_files[self._occ_index]}\t{masks}\t{decoys}\t{h_img}\n")
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
