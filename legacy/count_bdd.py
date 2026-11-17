import json

with open("bdd100k_labels_images_train.json") as f:
    data = json.load(f)

total = after_bike = daytime = daytime_ge25 = 0

for item in data:
    is_daytime = item.get("attributes", {}).get("timeofday") == "daytime"
    
    bikes = []
    for label in item.get("labels", []):
        if label.get("category") == "bike":
            b = label.get("box2d", {})
            if b:
                bikes.append((b.get("x1",0), b.get("y1",0), b.get("x2",0), b.get("y2",0)))
    
    for label in item.get("labels", []):
        if label.get("category") != "rider":
            continue
        total += 1
        r = label.get("box2d", {})
        if not r:
            continue
        rx1, ry1, rx2, ry2 = r.get("x1",0), r.get("y1",0), r.get("x2",0), r.get("y2",0)
        
        has_bike = False
        for (bx1, by1, bx2, by2) in bikes:
            ix1 = max(rx1, bx1); iy1 = max(ry1, by1)
            ix2 = min(rx2, bx2); iy2 = min(ry2, by2)
            if ix2 > ix1 and iy2 > iy1:
                r_area = (rx2-rx1)*(ry2-ry1)
                b_area = (bx2-bx1)*(by2-by1)
                inter = (ix2-ix1)*(iy2-iy1)
                union = r_area + b_area - inter
                if union > 0 and inter/union >= 0.1:
                    has_bike = True
                    break
        
        if not has_bike:
            continue
        after_bike += 1
        
        if is_daytime:
            daytime += 1
            h = ry2 - ry1
            if h >= 25:
                daytime_ge25 += 1

print(f"Всего rider: {total}")
print(f"После проверки bike (IoU>=0.1): {after_bike}")
print(f"Дневные: {daytime}")
print(f"Дневные >= 25px: {daytime_ge25}")