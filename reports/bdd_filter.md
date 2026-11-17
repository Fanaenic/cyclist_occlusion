# BDD100K: отбор rider + bike

Сопоставление: `global_greedy`, IoU >= 0.1; timeofday = daytime; целевая рамка `union`, её высота >= 25 px.

## Счётчики после каждого шага

| часть | шаг | изображений | объектов |
|---|---|---|---|
| train | all_images | 69863 |  |
| train | rider | 3586 | 4517 |
| train | matched | 2186 | 2581 |
| train | daytime | 1644 | 1952 |
| train | min_height | 1607 | 1908 |
| val | all_images | 10000 |  |
| val | rider | 515 | 649 |
| val | matched | 304 | 359 |
| val | daytime | 229 | 279 |
| val | min_height | 223 | 267 |

## Сравнение правил сопоставления (число сопоставленных rider)

| часть | nonexclusive (введение) | best_only | global_greedy |
|---|---|---|---|
| train | 2597 | 2577 | 2581 |
| val | 362 | 359 | 359 |

## Спорные случаи

| часть | случай | число |
|---|---|---|
| train | bike_multi_rider | 49 |
| train | rider_multi_bike | 37 |
| train | rider_multi_bike_alt | 4 |
| train | target_much_larger_than_rider | 70 |
| val | bike_multi_rider | 7 |
| val | rider_multi_bike | 10 |
| val | target_much_larger_than_rider | 7 |

## Сравнение с введением

| часть | шаг | введение | сейчас | разница |
|---|---|---|---|---|
| train | rider | 4517 | 4517 | +0 |
| train | matched | 2597 | 2581 | -16 |
| train | daytime | 1965 | 1952 | -13 |
| train | min_height | 1807 | 1908 | +101 |

## occluded / truncated rider после последнего шага

| часть | occluded | truncated | объектов |
|---|---|---|---|
| train | False | False | 168 |
| train | False | True | 15 |
| train | True | False | 1637 |
| train | True | True | 88 |
| val | False | False | 21 |
| val | False | True | 3 |
| val | True | False | 230 |
| val | True | True | 13 |
