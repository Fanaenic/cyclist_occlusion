Первые черновые скрипты подсчёта (до структуры проекта). Пути в них указывают на старое
расположение данных и больше не работают. Оставлены для истории; рабочий код — в `src/`.

- `count.py` — KITTI Cyclist: total / occ0 / occ0 ≥ 25 px.
- `count1.py` — BDD100K val: rider с occluded=false и truncated=false.
- `count_bdd.py` — BDD100K train: rider → bike IoU ≥ 0.1 (неэксклюзивно) → daytime → ≥ 25 px.
