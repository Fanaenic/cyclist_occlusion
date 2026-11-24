"""Замер скорости обучения proposed на полном train каждого датасета."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from run_experiment import load, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="новое имя замера внутри runs/benchmarks")
    parser.add_argument("--dataset", choices=("kitti", "bdd"), action="append")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch", type=int, help="переопределить batch только для замера")
    args = parser.parse_args()
    if Path(args.tag).name != args.tag or args.tag in ("", ".", ".."):
        parser.error("--tag должен быть именем каталога без пути")
    if args.epochs < 2 or (args.batch is not None and args.batch < 1):
        parser.error("нужно минимум две эпохи и положительный batch")

    config = load("configs/train.yaml")
    masks = load("configs/masks.yaml")
    yolo = load("configs/yolo.yaml")
    if config.get("offline"):
        os.environ["YOLO_OFFLINE"] = "1"
    if args.batch is not None:
        config["batch"] = args.batch
    datasets = args.dataset or config["datasets"]
    out = Path(config["out_dir"]) / "benchmarks" / args.tag
    if out.exists():
        parser.error(f"{out} уже существует; укажите новый --tag")
    out.mkdir(parents=True)
    for dataset in datasets:
        data = Path(yolo["out_dir"]) / f"{dataset}.yaml"
        print(f"{dataset}: полный train, proposed, batch={config['batch']}, эпох={args.epochs}", flush=True)
        started = perf_counter()
        train(config, masks, data, dataset, "proposed", config["seeds"][0], args.epochs, out)
        result = {"dataset": dataset, "batch": config["batch"], "epochs": args.epochs,
                  "workers": config["workers"], "total_seconds": round(perf_counter() - started, 1)}
        (out / f"{dataset}_timing.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
