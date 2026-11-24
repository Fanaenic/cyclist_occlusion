# Примеры конвертации: исходная запись -> YOLO

Формат YOLO: `класс cx cy w h`, координаты центра и размеры нормированы на ширину и высоту кадра.

## bdd

**1845d463-9a256fff.jpg** (1280×720)

Исходная запись:
```
{"id": 262939, "category": "rider", "box2d": {"x1": 1032.749638, "y1": 337.499882, "x2": 1081.349622, "y2": 413.099855}, "attributes": {"occluded": true, "truncated": false}}
{"id": 262941, "category": "bike", "box2d": {"x1": 1007.099648, "y1": 360.449874, "x2": 1081.349622, "y2": 422.549851}, "attributes": {"occluded": true, "truncated": false}}
```
YOLO:
```
0 0.815800 0.527812 0.058008 0.118125
```

**1ff92f74-697a077e.jpg** (1280×720)

Исходная запись:
```
{"id": 348191, "category": "rider", "box2d": {"x1": 929.562203, "y1": 248.917008, "x2": 962.906602, "y2": 310.126108}, "attributes": {"occluded": true, "truncated": false}}
{"id": 348219, "category": "bike", "box2d": {"x1": 923.084717, "y1": 271.768403, "x2": 966.047921, "y2": 321.143745}, "attributes": {"occluded": true, "truncated": false}}
```
YOLO:
```
0 0.737942 0.395876 0.033565 0.100315
```

**49b0144c-de5b3e1b.jpg** (1280×720)

Исходная запись:
```
{"id": 798802, "category": "rider", "box2d": {"x1": 561.929345, "y1": 487.346492, "x2": 570.006912, "y2": 506.194147}, "attributes": {"occluded": false, "truncated": false}}
{"id": 798803, "category": "bike", "box2d": {"x1": 561.390849, "y1": 499.462842, "x2": 569.468415, "y2": 514.271713}, "attributes": {"occluded": false, "truncated": false}}
```
YOLO:
```
0 0.441952 0.695568 0.006731 0.037396
```

**6309fc16-b2c13816.jpg** (1280×720)

Исходная запись:
```
{"id": 1077251, "category": "rider", "box2d": {"x1": 463.397829, "y1": 339.11559, "x2": 509.235799, "y2": 414.336876}, "attributes": {"occluded": true, "truncated": false}}
{"id": 1077252, "category": "bike", "box2d": {"x1": 468.099161, "y1": 392.005557, "x2": 506.885136, "y2": 433.142196}, "attributes": {"occluded": true, "truncated": false}}
```
YOLO:
```
0 0.379935 0.536290 0.035811 0.130593
```

**8fa5dbaa-c9aa57fe.jpg** (1280×720)

Исходная запись:
```
{"id": 1561690, "category": "rider", "box2d": {"x1": 509.304748, "y1": 315.298361, "x2": 549.228892, "y2": 448.890688}, "attributes": {"occluded": true, "truncated": false}}
{"id": 1561689, "category": "bike", "box2d": {"x1": 500.091485, "y1": 355.222504, "x2": 558.442157, "y2": 471.923848}, "attributes": {"occluded": true, "truncated": false}}
```
YOLO:
```
0 0.413490 0.546682 0.045586 0.217535
```

