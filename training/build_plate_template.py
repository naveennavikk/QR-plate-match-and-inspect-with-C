"""Build the 'average plate' used to isolate plate-specific marks (nails, damage, dirt, writing).

All plates share one printed design. Averaging many reference photos in the
shared QR-unit frame gives that common design; bgcontext subtracts it so only
what is unique to one physical plate is compared.

Usage: python training/build_plate_template.py [count]
Writes models/plate_template.npy
"""
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bgcontext import plate_frame_gray  # noqa: E402
from matcher import HybridImageMatcher  # noqa: E402

SOURCES = [Path.home() / "Documents" / n for n in ("Accept", "Acc")]
# Keep the evaluation plates (record.py sample, seed 11 from Acc) out of the template.
EXCLUDE_SEED, EXCLUDE_COUNT = 11, 100


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    acc = Path.home() / "Documents" / "Acc"
    acc_plates = sorted(p for p in acc.glob("*.jpeg") if not p.stem.lower().endswith("-house")
                        and p.with_name(p.stem + "-House.jpeg").exists())
    random.seed(EXCLUDE_SEED)
    excluded = {p.stem.casefold() for p in random.sample(acc_plates, EXCLUDE_COUNT)}
    pool = sorted({p.stem.casefold(): p for src in SOURCES if src.is_dir() for p in src.glob("*.jpeg")
                   if not p.stem.lower().endswith("-house") and p.stem.casefold() not in excluded}.values())
    random.seed(5)
    random.shuffle(pool)
    matcher = HybridImageMatcher()
    frames = []
    for path in pool:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        boxes = matcher._find_qr_boxes(rgb)
        if not boxes:
            continue
        x1, y1, x2, y2 = boxes[0]
        if min(x2 - x1, y2 - y1) < 60:   # only clear close-ups
            continue
        frames.append(plate_frame_gray(rgb, boxes[0]))
        if len(frames) >= count:
            break
    template = np.median(np.stack(frames), axis=0).astype(np.float32)
    out = ROOT / "models" / "plate_template.npy"
    np.save(out, template)
    print(f"Built template from {len(frames)} plates -> {out}")


if __name__ == "__main__":
    main()
