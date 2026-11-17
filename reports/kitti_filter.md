# KITTI: отбор Cyclist

occlusion = 0; truncation <= 0.05; высота >= 25 px. Подмножество occlusion = 1 — с теми же порогами truncation и высоты.

| подмножество | шаг | изображений | объектов |
|---|---|---|---|
| occlusion=0 | all_images | 7481 |  |
| occlusion=0 | cyclist | 1141 | 1627 |
| occlusion=0 | occlusion | 773 | 1010 |
| occlusion=0 | truncation | 716 | 925 |
| occlusion=0 | min_height | 640 | 832 |
| occlusion=1 | all_images | 7481 |  |
| occlusion=1 | cyclist | 1141 | 1627 |
| occlusion=1 | occlusion | 226 | 255 |
| occlusion=1 | truncation | 215 | 240 |
| occlusion=1 | min_height | 211 | 236 |

## Сравнение с введением

| шаг | введение | сейчас | разница |
|---|---|---|---|
| cyclist | 1627 | 1627 | +0 |
| occlusion | 1010 | 1010 | +0 |
| truncation | 925 | 925 | +0 |
| min_height | 832 | 832 | +0 |

## Распределение occlusion у всех Cyclist

| occlusion | объектов |
|---|---|
| 0 | 1010 |
| 1 | 255 |
| 2 | 80 |
| 3 | 282 |
