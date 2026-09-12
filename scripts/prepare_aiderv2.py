"""Turn the Kaggle AIDERv2 download into the layout the configs expect.

Download (needs a Kaggle API token in ~/.kaggle/kaggle.json):

    pip install kaggle
    kaggle datasets download -d banasmitajena/aiderv2-dataset
    unzip aiderv2-dataset.zip -d raw_aiderv2

Then:

    python scripts/prepare_aiderv2.py --src raw_aiderv2 --dst data/aiderv2 --resize 256

Output:  data/aiderv2/<split>/<class>/000123.jpg   with split in {train,val,test}

The official split is preserved -- AIDERv2 ships 13,399/1,670/1,654 and
re-splitting it would make our numbers incomparable with published results on
the same corpus.

Optional `--simulate-domains` additionally writes
    data/aiderv2_shift/<split>/<domain>/<class>/000123.jpg
by *partitioning* (never duplicating) each class across simulated acquisition
conditions. That gives a domain-incremental stream from a corpus with no event
metadata. It is a simulation of sensor/illumination shift and must be reported
as such -- it is not a substitute for real multi-event data like xBD.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import Counter
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# AIDERv2 folder names -> our canonical labels.
CLASS_ALIASES = {
    "earthquake": "earthquake", "earthquakes": "earthquake",
    "collapsed_building": "earthquake", "collapsed building": "earthquake",
    "flood": "flood", "floods": "flood", "flooded": "flood",
    "fire": "fire", "fires": "fire", "wildfire": "fire", "fire_smoke": "fire",
    "normal": "normal", "none": "normal", "non_disaster": "normal",
}

SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "val": "val", "valid": "val", "validation": "val",
    "test": "test", "testing": "test",
}

# name -> (colour gain RGB, haze, blur radius, downscale factor, noise sd)
SIM_DOMAINS = {
    "clear_nadir":   ((1.00, 1.00, 1.00), 0.00, 0.0, 1.0, 0.00),
    "dusk_haze":     ((1.12, 0.98, 0.86), 0.18, 1.2, 1.0, 0.02),
    "lowres_oblique": ((0.94, 0.97, 1.08), 0.06, 0.4, 0.5, 0.04),
}


def find_image_root(src: Path) -> Path:
    """Locate the directory whose children are the split folders."""
    candidates = []
    for d in [src, *sorted(p for p in src.rglob("*") if p.is_dir())]:
        children = {c.name.lower() for c in d.iterdir() if c.is_dir()}
        if children and children <= set(SPLIT_ALIASES):
            candidates.append(d)
    if not candidates:
        raise SystemExit(
            f"Could not find Train/Validation/Test folders under {src}. "
            "Pass --src pointing at the unzipped dataset."
        )
    return min(candidates, key=lambda p: len(p.parts))


def canonical_class(name: str) -> str | None:
    return CLASS_ALIASES.get(name.strip().lower().replace("-", "_"))


def assign_domain(path: Path, domains: list[str]) -> str:
    """Stable hash-based assignment so the partition is reproducible."""
    h = hashlib.md5(path.name.encode()).hexdigest()
    return domains[int(h, 16) % len(domains)]


def transform_domain(img, spec):
    from PIL import Image, ImageFilter
    import numpy as np

    gain, haze, blur, scale, noise = spec
    if scale != 1.0:
        w, h = img.size
        small = img.resize((max(8, int(w * scale)), max(8, int(h * scale))), Image.BILINEAR)
        img = small.resize((w, h), Image.BILINEAR)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = arr * np.array(gain, dtype=np.float32)
    arr = arr * (1 - haze) + haze * 0.80
    if noise > 0:
        arr = arr + np.random.default_rng(abs(hash(img.size)) % 2**32).normal(0, noise, arr.shape)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))


def load_and_save(src_path: Path, dst_path: Path, resize: int | None, spec=None) -> bool:
    from PIL import Image

    try:
        with Image.open(src_path) as im:
            img = im.convert("RGB")
            if resize:
                w, h = img.size
                if max(w, h) > resize:
                    ratio = resize / max(w, h)
                    img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))),
                                     Image.BILINEAR)
            if spec is not None:
                img = transform_domain(img, spec)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, quality=92)
        return True
    except Exception as exc:
        print(f"[skip] {src_path}: {exc}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="unzipped AIDERv2 download")
    ap.add_argument("--dst", default="data/aiderv2")
    ap.add_argument("--resize", type=int, default=256,
                    help="longest side in px; 256 cuts training time a lot with no accuracy cost "
                         "at 224 input. 0 disables.")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="cap images per class per split (for quick debugging)")
    ap.add_argument("--simulate-domains", action="store_true",
                    help="also write a domain-incremental variant at <dst>_shift")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} exists; pass --overwrite to rebuild")
        shutil.rmtree(dst)

    image_root = find_image_root(src)
    print(f"found image root: {image_root}")

    counts: Counter = Counter()
    unknown: Counter = Counter()
    shift_dst = Path(str(dst) + "_shift")
    if args.simulate_domains and shift_dst.exists() and args.overwrite:
        shutil.rmtree(shift_dst)

    for split_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        split = SPLIT_ALIASES.get(split_dir.name.lower())
        if split is None:
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            cls = canonical_class(class_dir.name)
            if cls is None:
                unknown[class_dir.name] += 1
                continue
            files = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
            if args.limit_per_class:
                files = files[: args.limit_per_class]
            for i, f in enumerate(files):
                # Rename: the originals are "img (1).jpg" -- spaces and
                # parentheses break shell globbing and some dataloaders.
                out = dst / split / cls / f"{i:06d}.jpg"
                if load_and_save(f, out, args.resize or None):
                    counts[(split, cls)] += 1
                if args.simulate_domains:
                    dom = assign_domain(f, list(SIM_DOMAINS))
                    out2 = shift_dst / split / dom / cls / f"{i:06d}.jpg"
                    load_and_save(f, out2, args.resize or None, SIM_DOMAINS[dom])

    if unknown:
        print(f"[warn] unrecognised class folders (ignored): {dict(unknown)}")
    if not counts:
        raise SystemExit("No images written -- check --src")

    splits = sorted({s for s, _ in counts})
    classes = sorted({c for _, c in counts})
    print("\n" + "class".ljust(14) + "".join(s.rjust(10) for s in splits) + "total".rjust(10))
    for c in classes:
        row = [counts[(s, c)] for s in splits]
        print(c.ljust(14) + "".join(str(v).rjust(10) for v in row) + str(sum(row)).rjust(10))
    total_row = [sum(counts[(s, c)] for c in classes) for s in splits]
    print("TOTAL".ljust(14) + "".join(str(v).rjust(10) for v in total_row)
          + str(sum(total_row)).rjust(10))
    print(f"\nwrote {dst}")
    if args.simulate_domains:
        print(f"wrote {shift_dst} (simulated acquisition domains: {list(SIM_DOMAINS)})")


if __name__ == "__main__":
    main()
