# Влияние степени окклюзии на качество детекции велосипедистов
Детектор YOLOv8n с одним классом Cyclist обучается на KITTI и BDD100K.
Затем проверяется, как падает Recall, когда рамку объекта закрывает синтетическая маска на 0/30/50/70/90 % площади.
Сравниваются режимы обучения: baseline (без окклюзионной аугментации), proposed (маски на заданную долю рамки),
ORE (Random Erasing внутри рамки) и абляция proposed_noise (шумовая заливка без приманок).

## Установка

Нужны Python 3.12 и Git. Видеокарта NVIDIA желательна (torch собирается под CUDA 12.6).

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; в Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Предобученные веса YOLOv8n (COCO) положить в `weights/yolov8n.pt`:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt (SHA-256 — в `configs/train.yaml`).

## Данные

Данные в репозиторий не входят. Их нужно разложить так:

```
data/
├── kitti/
│   ├── label_2/                 # метки KITTI object (training), 7481 файл
│   ├── image_2/                 # изображения KITTI object (training)
│   └── devkit/mapping/          # train_mapping.txt, train_rand.txt из devkit_object.zip
└── bdd100k/
    ├── labels/                  # bdd100k_labels_images_{train,val}.json
    └── images/{train,val}/      # 100K images, все .jpg в одной папке без подпапок
```

## Запуск по шагам

Все команды выполняются из корня проекта. Параметры лежат в `configs/`, отчёты пишутся в `reports/`.

```bash
python src/kitti_filter.py                  # 1. отбор Cyclist в KITTI
python src/bdd_filter.py                    # 1. отбор rider + bike в BDD100K
python src/make_splits.py                   # 2. train / calib / test по кластерам + проверка пересечений
python src/masks.py generate                # 3. тестовые маски и контрольные заплатки (один раз)
python src/masks.py check                   #    проверка доли перекрытия ±2 п.п.
python src/masks.py placement               #    положение приманок и контроля относительно целей
python src/masks.py examples                #    картинки уровней масок
python src/to_yolo.py                       # 4. разметка в формате YOLO + ignore-области
python src/evaluate.py --population-report  # 5. совокупность целей для кривой деградации
python src/power_map.py                     # 6. синтетика для условия mAP (до обучения)
python src/power_recall.py                  #    синтетика: мощность проверки Recall (до обучения)
python src/check_workers.py                 #    воспроизводимость масок обучения при workers > 0
python -m pytest -q tests                   # юнит-тесты
```

Обучение и оценка (обучение → порог на calib → test на всех уровнях масок → бутстрэп):

```bash
python src/run_experiment.py --smoke                 # технический прогон: 200 изображений, 3 эпохи
python src/run_experiment.py --tag main              # основная серия
```



## Структура

```
.
├── configs/                 # все параметры
│   ├── bdd_filter.yaml      #   отбор BDD100K: rider ∪ bike, IoU, daytime, высота
│   ├── kitti_filter.yaml    #   отбор KITTI
│   ├── splits.yaml          #   разбиение и кластеры
│   ├── masks.yaml           #   маски, заливка, приманки, режимы обучения
│   ├── yolo.yaml            #   конвертация в YOLO
│   ├── train.yaml           #   обучение YOLOv8n
│   └── protocol.yaml        #   NMS, сопоставление, FPPI, статистика, правило решения
├── src/
│   ├── kitti_filter.py      #   отбор KITTI
│   ├── bdd_filter.py        #   отбор BDD100K, сопоставление rider ↔ bike
│   ├── make_splits.py       #   train / calib / test по кластерам
│   ├── masks.py             #   генератор масок, приманки, проверки, примеры
│   ├── to_yolo.py           #   разметка YOLO + ignore
│   ├── occlusion_train.py   #   маски при обучении Ultralytics
│   ├── evaluate.py          #   оценщик: Recall при пороге, FPPI, AP@0.5
│   ├── stats.py             #   seed + кластерный бутстрэп, правило решения
│   ├── power_map.py         #   синтетика для условия mAP
│   ├── power_recall.py      #   синтетика: мощность проверки Recall
│   ├── check_workers.py     #   проверка воспроизводимости масок при workers > 0
│   └── run_experiment.py    #   весь конвейер обучения и оценки
├── tests/                   # юнит-тесты оценщика, масок, статистики
├── splits/                  # списки частей, кластеры, тестовые маски, контрольные заплатки
├── reports/                 # счётчики, проверки, отчёты прогонов
│   ├── examples/            #   картинки: уровни масок, приманки, рамки YOLO
│   └── smoke_history/       #   отчёты первых технических прогонов (до исправлений протокола)
├── requirements.txt
└── README.md
```

Не входят в репозиторий: `data/`, `datasets_yolo/` (собирается `src/to_yolo.py`), `runs/`, `weights/`.
