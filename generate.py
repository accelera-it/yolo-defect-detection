import cv2
import numpy as np
import os
import random
from pathlib import Path

IMG_SIZE = 640
TOTAL_IMAGES = 120
SPLIT = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}
BASE_DIR = Path("dataset")

def yolo_bbox(cx, cy, w, h):
    return (
        cx / IMG_SIZE,
        cy / IMG_SIZE,
        w / IMG_SIZE,
        h / IMG_SIZE
    )

# Create folders, removing images and labels left by an earlier run:
# the split is random, so old files would otherwise stay in other splits
for split in SPLIT:
    for kind, pattern in (("images", "*.jpg"), ("labels", "*.txt")):
        folder = BASE_DIR / kind / split
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(pattern):
            old.unlink()

def generate_image(idx, split):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    labels = []

    for _ in range(random.randint(1, 3)):
        obj = random.choice(["potato", "carrot"])

        if obj == "potato":
            cx, cy = random.randint(150, 500), random.randint(150, 500)
            r = random.randint(50, 80)
            cv2.circle(img, (cx, cy), r, (70, 120, 160), -1)

            if random.random() > 0.5:
                dx, dy = cx + 12, cy + 12
                dr = 14
                cv2.circle(img, (dx, dy), dr, (0, 0, 0), -1)
                cls = random.choice([1, 2])
                w = h = dr * 2
                x, y = dx, dy
            else:
                cls = 0
                w = h = r * 2
                x, y = cx, cy

        else:
            x1, y1 = random.randint(120, 420), random.randint(120, 420)
            w, h = random.randint(60, 90), random.randint(160, 220)
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 140, 255), -1)

            if random.random() > 0.5:
                cv2.line(img, (x1 + 10, y1 + 40),
                         (x1 + w - 10, y1 + h - 40), (0, 0, 0), 4)
                cls = random.choice([4, 5])
            else:
                cls = 3

            x, y = x1 + w // 2, y1 + h // 2

        labels.append((cls, *yolo_bbox(x, y, w, h)))

    name = f"img_{idx:04d}"
    cv2.imwrite(str(BASE_DIR / "images" / split / f"{name}.jpg"), img)

    with open(BASE_DIR / "labels" / split / f"{name}.txt", "w") as f:
        for l in labels:
            f.write(" ".join(map(str, l)) + "\n")

# Generate dataset
indices = list(range(TOTAL_IMAGES))
random.shuffle(indices)

train_end = int(TOTAL_IMAGES * SPLIT["train"])
val_end = train_end + int(TOTAL_IMAGES * SPLIT["val"])

splits = {
    "train": indices[:train_end],
    "val": indices[train_end:val_end],
    "test": indices[val_end:]
}

for split, idxs in splits.items():
    for idx in idxs:
        generate_image(idx, split)

print("Dataset: train/val/test split generated...")
