"""Подготовка изображений и листа ручной проверки спорных пар BDD100K."""
import csv
import json
from pathlib import Path

import cv2
import yaml

from bdd_filter import box_of, target_box


def draw(image, box, label, color):
    x1, y1, x2, y2 = map(round, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


def main():
    config = yaml.safe_load(Path("configs/bdd_filter.yaml").read_text(encoding="utf-8"))
    reports = Path(config["out_dir"])
    output = reports / "rider_bike_review"
    output.mkdir(parents=True, exist_ok=True)
    with (reports / "bdd_ambiguous.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    review = output / "review.csv"
    previous = {}
    fields = list(rows[0]) + ["panel", "decision", "comment"]
    if review.exists():
        with review.open(encoding="utf-8-sig", newline="") as fh:
            previous = {tuple(row[key] for key in fields[:6]): row for row in csv.DictReader(fh)}
    for split in config["splits"]:
        cases = [(i, row) for i, row in enumerate(rows) if row["split"] == split]
        names = {row["image"] for _, row in cases}
        source = Path(config["labels_dir"]) / config["label_file"].format(split=split)
        frames = {frame["name"]: frame for frame in json.loads(source.read_text(encoding="utf-8"))
                  if frame["name"] in names}
        for i, row in cases:
            frame = frames[row["image"]]
            labels = {str(label["id"]): label for label in frame["labels"]}
            path = Path(config["labels_dir"]).parent / "images" / split / row["image"]
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"Не удалось прочитать {path}")
            anchor = row["object_id"]
            assigned = row["assigned"].split(" ", 1)[0]
            draw(image, box_of(labels[anchor]), f"{labels[anchor]['category']} {anchor}", (255, 180, 0))
            for item in filter(None, row["candidates"].split(";")):
                candidate, score = item.rsplit(":", 1)
                color = (0, 220, 0) if candidate == assigned else (0, 190, 255)
                draw(image, box_of(labels[candidate]), f"{candidate} IoU {score}", color)
            if assigned:
                a, b = box_of(labels[anchor]), box_of(labels[assigned])
                draw(image, target_box(a, b, "union"), "union", (255, 0, 255))
            row["panel"] = f"{i:04d}_{split}_{Path(row['image']).stem}.png"
            if not cv2.imwrite(str(output / row["panel"]), image):
                raise OSError(f"Не удалось сохранить {row['panel']}")
            old = previous.get(tuple(row[key] for key in fields[:6]), {})
            row["decision"], row["comment"] = old.get("decision", ""), old.get("comment", "")
    with review.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
