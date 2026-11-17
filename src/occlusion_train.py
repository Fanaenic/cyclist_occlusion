"""Окклюзионная аугментация при обучении Ultralytics: маски proposed / ORE на рамках целей.

Маска накладывается в OcclusionDataset.update_labels_info — на загруженное (масштабированное) изображение
до mosaic и прочих аугментаций Ultralytics; mosaic берёт свои плитки через тот же метод, поэтому маски
получают и они. Каждая цель маскируется независимо с вероятностью p режима; на каждую маску на фон
ставится такая же заплатка-приманка (masks.apply_training_masks, configs/masks.yaml).
Соседи и запретные для приманок области при обучении — только цели кадра (прочие классы в YOLO-разметке
не хранятся).
"""
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

from masks import apply_training_masks


class OcclusionDataset(YOLODataset):
    occ = None            # {"regime": "proposed" | "ore", "cfg": весь configs/masks.yaml, "seed": int}

    def get_image_and_label(self, index, full=False):
        self._occ_index = index
        return super().get_image_and_label(index, full)

    def update_labels_info(self, label):
        o = self.occ
        boxes = label.get("bboxes")
        if o and boxes is not None and len(boxes):
            self._occ_calls = getattr(self, "_occ_calls", 0) + 1
            rng = np.random.default_rng([o["seed"], self._occ_index, self._occ_calls])
            h, w = label["resized_shape"]
            px = []
            for cx, cy, bw, bh in boxes:                # пиксельная рамка, обрезанная по кадру
                x0 = min(max(int((cx - bw / 2) * w), 0), w - 1)
                y0 = min(max(int((cy - bh / 2) * h), 0), h - 1)
                x1 = min(max(x0 + 1, int(np.ceil((cx + bw / 2) * w))), w)
                y1 = min(max(y0 + 1, int(np.ceil((cy + bh / 2) * h))), h)
                px.append((x0, y0, x1, y1))
            label["img"], _, _ = apply_training_masks(label["img"], px, rng, o["regime"], o["cfg"])
        return super().update_labels_info(label)


class OcclusionTrainer(DetectionTrainer):
    occ = None            # задаётся перед model.train(trainer=OcclusionTrainer)

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode, batch)
        if mode == "train" and self.occ and self.occ["cfg"]["training"][self.occ["regime"]].get("p", 0) > 0:
            ds.__class__ = OcclusionDataset
            ds.occ = self.occ
        return ds
