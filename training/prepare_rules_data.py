"""Build the accept/reject rule classification dataset.

Sources (all read-only):
  * Documents/rejection rules data train/<rule>/   - new reject rule folders
  * Documents/Acc, Accept, Accept-, accept--       - accepted plate + "-House" pairs
  * the older prepared dataset the shipped models were trained on (optional)

Output: training/data/prepared/{train,val,test}/<class>/  plus
        training/data/holdout_accept/  (accept pairs never used for training,
        used by evaluate_rules.py to measure false rejections).

Images are split by house ID, so a plate photo and its property photo
never end up on both sides of the split.
"""
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
DOCS = Path.home() / "Documents"
REJECT_ROOT = DOCS / "rejection rules data train"
ACCEPT_DIRS = [DOCS / n for n in ("Accept", "Acc", "Accept-", "accept--")]
OLD_PREPARED = Path.home() / "Downloads" / "data train classification and yolo" / \
    "QR_House_Plate_AI_Inspector_Hybrid_v2" / "data" / "prepared"
OUT = HERE / "data" / "prepared"
HOLDOUT = HERE / "data" / "holdout_accept"

SEED = 42
ACCEPT_CLASS = "Accept+"
ACCEPT_TRAIN_GROUPS = 600      # ~1200 accept images (plate + house) for training
HOLDOUT_GROUPS = 300           # accept pairs reserved for false-reject testing
MAX_SIDE = 640
EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".bmp", ".webp"}

# Folders that describe the same rule are trained as one class.
CLASS_ALIASES = {
    "write r1,r2 or o1, o2 on plate": "Write O for owner and R1, R2 for Rent",
    "rescan properly scanify image": "Rescan Scanify Image",
    "accept+": ACCEPT_CLASS,
}

ID_RE = re.compile(r"(hpsba-[a-z]?\d+)", re.I)


def canonical(name):
    return CLASS_ALIASES.get(" ".join(name.split()).casefold(), " ".join(name.split()))


def group_key(path, class_name):
    m = ID_RE.search(path.stem)
    return m.group(1).casefold() if m else f"{class_name}/{path.stem}".casefold()


def images_in(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_jpg(src, dest):
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, "JPEG", quality=92)


def split_groups(groups):
    """Split a list of image groups ~70/15/15, keeping each group intact."""
    random.shuffle(groups)
    n = len(groups)
    if n >= 5:
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
    elif n >= 3:
        n_val, n_test = 1, 1
    elif n == 2:
        n_val, n_test = 1, 0
    else:
        n_val, n_test = 0, 0
    return {
        "val": groups[:n_val],
        "test": groups[n_val:n_val + n_test],
        "train": groups[n_val + n_test:],
    }


def main():
    random.seed(SEED)
    if not REJECT_ROOT.is_dir():
        raise FileNotFoundError(REJECT_ROOT)

    # 1. Collect reject images: new rule folders first, then old classes.
    class_files = {}   # class -> {hash: path}
    sources = [d for d in sorted(REJECT_ROOT.iterdir()) if d.is_dir()]
    if OLD_PREPARED.is_dir():
        for split in ("train", "val", "test"):
            sources += [d for d in sorted((OLD_PREPARED / split).iterdir()) if d.is_dir()]
    seen = {}          # hash -> class (first wins; drops cross-class duplicates)
    conflicts = 0
    for folder in sources:
        cls = canonical(folder.name)
        for img in images_in(folder):
            h = digest(img)
            if h in seen:
                conflicts += seen[h] != cls
                continue
            seen[h] = cls
            class_files.setdefault(cls, {})[h] = img

    reject_ids = {
        group_key(p, c)
        for c, files in class_files.items() if c != ACCEPT_CLASS
        for p in files.values()
        if ID_RE.search(p.stem)
    }

    # 2. Accept pairs from the Accept folders, excluding any house ID that was
    #    ever rejected and any exact duplicate of a reject image.
    accept_groups = {}
    for folder in ACCEPT_DIRS:
        if not folder.is_dir():
            continue
        for img in images_in(folder):
            key = group_key(img, ACCEPT_CLASS)
            if key in reject_ids:
                continue
            accept_groups.setdefault(key, {})[img.name.casefold()] = img
    # Only keep complete plate + house pairs.
    pairs = {k: list(v.values()) for k, v in accept_groups.items() if len(v) == 2}
    keys = sorted(pairs)
    random.shuffle(keys)
    holdout_keys = keys[:HOLDOUT_GROUPS]
    train_keys = keys[HOLDOUT_GROUPS:HOLDOUT_GROUPS + ACCEPT_TRAIN_GROUPS]
    accept_files = class_files.setdefault(ACCEPT_CLASS, {})
    for k in train_keys:
        for img in pairs[k]:
            h = digest(img)
            if h not in seen:
                seen[h] = ACCEPT_CLASS
                accept_files[h] = img

    # 3. Write the split dataset.
    for target in (OUT, HOLDOUT):
        if target.exists():
            shutil.rmtree(target)
    summary = {}
    for cls, files in sorted(class_files.items()):
        grouped = {}
        for img in files.values():
            grouped.setdefault(group_key(img, cls), []).append(img)
        splits = split_groups(list(grouped.values()))
        summary[cls] = {}
        for split, groups in splits.items():
            count = 0
            for group in groups:
                for img in group:
                    count += 1
                    name = re.sub(r"[^\w.-]+", "_", img.stem)[:60]
                    save_jpg(img, OUT / split / cls / f"{name}_{digest(img)[:8]}.jpg")
            summary[cls][split] = count
    for k in holdout_keys:
        for img in pairs[k]:
            kind = "house" if img.stem.casefold().endswith("-house") else "plate"
            save_jpg(img, HOLDOUT / kind / f"{img.stem}.jpg")

    report = {
        "seed": SEED,
        "classes": summary,
        "cross_class_duplicates_dropped": conflicts,
        "accept_pairs_available": len(pairs),
        "accept_pairs_trained": len(train_keys),
        "accept_pairs_holdout": len(holdout_keys),
    }
    (HERE / "reports").mkdir(exist_ok=True)
    (HERE / "reports" / "split_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
