import json

with open("bdd100k_labels_images_val.json") as f:
    data = json.load(f)

total = after_occ = daytime = daytime_ge25 = 0
for item in data:
    is_daytime = item.get("attributes", {}).get("timeofday") == "daytime"
    for label in item.get("labels", []):
        if label.get("category") != "rider":
            continue
        total += 1
        attrs = label.get("attributes", {})
        if not attrs.get("occluded") and not attrs.get("truncated"):
            after_occ += 1
            if is_daytime:
                daytime += 1
                h = label.get("box2d", {}).get("y2", 0) - label.get("box2d", {}).get("y1", 0)
                if h >= 25:
                    daytime_ge25 += 1

print(f"Всего rider в val: {total}")
print(f"После occluded=false и truncated=false: {after_occ}")
print(f"Дневные: {daytime}")
print(f"Дневные >= 25px: {daytime_ge25}")