import glob

counts = {"total": 0, "occ0": 0, "occ0_trunc005": 0, "occ0_ge25px": 0, "occ0_trunc005_ge25px": 0}
for f in glob.glob("training/label_2/*.txt"):
    for line in open(f):
        p = line.split()
        if p[0] != "Cyclist":
            continue
        counts["total"] += 1
        if int(p[2]) == 0:
            counts["occ0"] += 1
            if float(p[1]) <= 0.05:
                counts["occ0_trunc005"] += 1
                if float(p[7]) - float(p[5]) >= 25:
                    counts["occ0_trunc005_ge25px"] += 1
            if float(p[7]) - float(p[5]) >= 25:
                counts["occ0_ge25px"] += 1
print(counts)