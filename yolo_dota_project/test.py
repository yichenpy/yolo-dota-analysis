import os

label_dir = "/workspace/cy/yolo_dota_project/datasets/DOTA-split/labels/train"
bad = []

for f in os.listdir(label_dir):
    if not f.endswith(".txt"):
        continue
    p = os.path.join(label_dir, f)
    with open(p, "r") as fp:
        for line in fp:
            sp = line.strip().split()
            if len(sp) != 9:  # DOTA OBB 格式必须是 class cx cy w h theta x1 y1 x2 y2 之类，不同版本略有差异
                bad.append(p)
                break
print("Bad label files:", bad)
