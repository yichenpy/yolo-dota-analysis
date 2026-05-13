from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import cv2

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from ultralytics.data.split_dota import get_window_obj, get_windows, load_yolo_dota, split_test


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_data_root = project_root / "datasets" / "DOTA"
    default_save_dir = project_root / "datasets" / "DOTA-split-lite"
    default_source_yaml = default_data_root / "data.yaml"
    default_size_summary = project_root / "analysis" / "outputs" / "object_sizes" / "object_size_summary.json"

    parser = argparse.ArgumentParser(
        description="Split DOTA images into train/val/test tiles with optional background filtering."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="Original DOTA dataset root.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=default_save_dir,
        help="Output directory for the sliced dataset. Use a new directory to avoid overwriting prior splits.",
    )
    parser.add_argument(
        "--source-yaml",
        type=Path,
        default=default_source_yaml,
        help="Original dataset yaml used to copy class names into the generated sliced-data yaml.",
    )
    parser.add_argument(
        "--size-summary",
        type=Path,
        default=default_size_summary,
        help="Path to object_size_summary.json used to auto-recommend overlap and background retention.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Dataset splits to generate. Default skips test because it is not needed for training.",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=[1.0],
        help="Scale rates for multi-scale slicing. Using only 1.0 is the simplest way to shrink the dataset.",
    )
    parser.add_argument("--crop-size", type=int, default=1024, help="Base crop size before rate scaling.")
    parser.add_argument(
        "--gap",
        type=int,
        default=None,
        help="Overlap between neighboring tiles. If omitted, the script recommends a value from object_size_summary.json.",
    )
    parser.add_argument(
        "--iof-thr",
        type=float,
        default=0.7,
        help="IoF threshold used by Ultralytics when assigning objects to windows.",
    )
    parser.add_argument(
        "--im-rate-thr",
        type=float,
        default=0.6,
        help="Minimum valid-image ratio for a crop window. Keep the Ultralytics default unless you have a reason to change it.",
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=1,
        help="Minimum number of assigned objects required to keep a tile. Default 1 drops empty tiles.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow keeping empty/background tiles. Disabled by default to reduce dataset size.",
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=None,
        help="When --allow-empty is set, keep at most this many empty tiles per positive tile, per source image. If omitted, the script recommends a value from object_size_summary.json.",
    )
    parser.add_argument(
        "--max-background-per-image",
        type=int,
        default=None,
        help="Hard cap on empty tiles kept from each source image when --allow-empty is set. If omitted, the script recommends a value from object_size_summary.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling kept background tiles.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    return parser.parse_args()


def iter_with_progress(items: list[dict], desc: str, disable: bool) -> Iterable[dict]:
    if tqdm is None:
        return items
    return tqdm(items, desc=desc, unit="image", disable=disable)


def load_names_from_yaml(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    in_names = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped == "names:":
            in_names = True
            continue
        if in_names:
            if not raw_line.startswith("  "):
                break
            key, value = stripped.split(":", 1)
            names[int(key.strip())] = value.strip().strip("'\"")
    if not names:
        raise ValueError(f"Failed to parse class names from yaml: {path}")
    return names


def write_split_data_yaml(save_dir: Path, source_yaml: Path, splits: list[str]) -> Path:
    names = load_names_from_yaml(source_yaml)
    lines = [f"path: {save_dir.as_posix()}"]
    for split in ("train", "val", "test"):
        if split in splits:
            lines.append(f"{split}: images/{split}")
    lines.append("")
    lines.append("names:")
    for class_id, class_name in sorted(names.items()):
        lines.append(f"  {class_id}: {class_name}")

    output_yaml = save_dir / "data.yaml"
    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_yaml


def recommend_split_settings(size_summary_path: Path, crop_size: int, rates: list[float]) -> dict[str, float | int | str]:
    if not size_summary_path.exists():
        return {
            "profile": "fallback",
            "median": -1.0,
            "q75": -1.0,
            "q90": -1.0,
            "gap": 200,
            "background_ratio": 0.05,
            "max_background_per_image": 4,
            "reason": "size summary not found; using conservative defaults for small-object aerial detection",
        }

    summary = json.loads(size_summary_path.read_text(encoding="utf-8"))
    median = float(summary.get("median", 0.0))
    q75 = float(summary.get("q75", 0.0))
    q90 = float(summary.get("q90", 0.0))

    if median <= 32.0 and q75 <= 48.0:
        recommendation = {
            "profile": "small-object-heavy",
            "gap": 200,
            "background_ratio": 0.05,
            "max_background_per_image": 4,
            "reason": "median and upper quartile are both in the small-object regime; keep moderate overlap and only a small background fraction",
        }
    elif median <= 48.0 and q75 <= 80.0:
        recommendation = {
            "profile": "mixed-small-medium",
            "gap": 160,
            "background_ratio": 0.08,
            "max_background_per_image": 6,
            "reason": "objects are still relatively small, but not as concentrated in the smallest bins",
        }
    else:
        recommendation = {
            "profile": "medium-large",
            "gap": 128,
            "background_ratio": 0.10,
            "max_background_per_image": 8,
            "reason": "larger objects need less overlap, and the dataset can tolerate a slightly larger background sample",
        }

    if len(rates) > 1:
        recommendation["background_ratio"] = min(float(recommendation["background_ratio"]), 0.05)
        recommendation["reason"] += "; multi-scale slicing already expands the tile count, so background retention is capped more aggressively"

    recommendation["median"] = median
    recommendation["q75"] = q75
    recommendation["q90"] = q90
    recommendation["gap"] = min(int(recommendation["gap"]), max(64, crop_size // 4))
    return recommendation


def resolve_split_settings(args: argparse.Namespace) -> dict[str, float | int | str]:
    recommendation = recommend_split_settings(args.size_summary, args.crop_size, args.rates)

    if args.gap is None:
        args.gap = int(recommendation["gap"])
    if args.background_ratio is None:
        args.background_ratio = float(recommendation["background_ratio"])
    if args.max_background_per_image is None:
        args.max_background_per_image = int(recommendation["max_background_per_image"])
    if not args.allow_empty and args.background_ratio > 0 and args.max_background_per_image != 0:
        args.allow_empty = True
        recommendation["auto_enabled_background"] = True
    else:
        recommendation["auto_enabled_background"] = False

    return recommendation


def select_keep_indices(
    window_objs: list,
    *,
    min_objects: int,
    allow_empty: bool,
    background_ratio: float,
    max_background_per_image: int,
    rng: random.Random,
) -> tuple[list[int], int, int]:
    positive_indices = [i for i, objs in enumerate(window_objs) if len(objs) >= min_objects]
    background_indices = [i for i, objs in enumerate(window_objs) if len(objs) == 0]

    if not allow_empty or not background_indices:
        return sorted(positive_indices), len(positive_indices), 0

    limits = [len(background_indices)]
    if background_ratio >= 0:
        limits.append(int(round(len(positive_indices) * background_ratio)))
    if max_background_per_image >= 0:
        limits.append(max_background_per_image)

    keep_background = max(0, min(limits))
    if keep_background == 0:
        return sorted(positive_indices), len(positive_indices), 0

    chosen_background = rng.sample(background_indices, k=min(keep_background, len(background_indices)))
    keep_indices = sorted(positive_indices + chosen_background)
    return keep_indices, len(positive_indices), len(chosen_background)


def crop_and_save_filtered(
    anno: dict,
    windows,
    window_objs: list,
    keep_indices: list[int],
    im_dir: Path,
    lb_dir: Path,
) -> int:
    image = cv2.imread(str(anno["filepath"]))
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {anno['filepath']}")

    base_name = Path(anno["filepath"]).stem
    saved = 0
    for index in keep_indices:
        x_start, y_start, x_stop, y_stop = windows[index].tolist()
        patch = image[y_start:y_stop, x_start:x_stop]
        if patch.size == 0:
            continue
        patch_h, patch_w = patch.shape[:2]

        patch_name = f"{base_name}__{x_stop - x_start}__{x_start}___{y_start}"
        cv2.imwrite(str(im_dir / f"{patch_name}.jpg"), patch)

        labels = window_objs[index].copy()
        if len(labels):
            labels[:, 1::2] -= x_start
            labels[:, 2::2] -= y_start
            labels[:, 1::2] = labels[:, 1::2].clip(0, patch_w) / max(patch_w, 1)
            labels[:, 2::2] = labels[:, 2::2].clip(0, patch_h) / max(patch_h, 1)
            with (lb_dir / f"{patch_name}.txt").open("w", encoding="utf-8") as f:
                for row in labels:
                    class_id = str(int(round(float(row[0]))))
                    coords = " ".join(f"{float(value):.6f}".rstrip("0").rstrip(".") for value in row[1:])
                    f.write(f"{class_id} {coords}\n")
        else:
            (lb_dir / f"{patch_name}.txt").touch()
        saved += 1
    return saved


def split_labeled_split(args: argparse.Namespace, split: str, rng: random.Random) -> None:
    crop_sizes = [int(args.crop_size / rate) for rate in args.rates]
    gaps = [int(args.gap / rate) for rate in args.rates]

    save_root = args.save_dir
    im_dir = save_root / "images" / split
    lb_dir = save_root / "labels" / split
    im_dir.mkdir(parents=True, exist_ok=True)
    lb_dir.mkdir(parents=True, exist_ok=True)

    annos = load_yolo_dota(data_root=str(args.data_root), split=split)
    total_windows = 0
    positive_windows = 0
    kept_background = 0
    kept_windows = 0

    iterator = iter_with_progress(annos, f"split {split}", args.no_progress)
    for anno in iterator:
        windows = get_windows(
            im_size=anno["ori_size"],
            crop_sizes=crop_sizes,
            gaps=gaps,
            im_rate_thr=args.im_rate_thr,
        )
        window_objs = get_window_obj(anno, windows, args.iof_thr)
        keep_indices, positive_count, background_count = select_keep_indices(
            window_objs,
            min_objects=args.min_objects,
            allow_empty=args.allow_empty,
            background_ratio=args.background_ratio,
            max_background_per_image=args.max_background_per_image,
            rng=rng,
        )

        total_windows += len(windows)
        positive_windows += positive_count
        kept_background += background_count
        kept_windows += crop_and_save_filtered(anno, windows, window_objs, keep_indices, im_dir, lb_dir)

    print(
        f"[{split}] total_windows={total_windows} "
        f"positive_windows={positive_windows} "
        f"kept_background={kept_background} "
        f"saved_tiles={kept_windows}"
    )


def split_test_split(args: argparse.Namespace) -> None:
    split_test(
        data_root=str(args.data_root),
        save_dir=str(args.save_dir),
        rates=args.rates,
        crop_size=args.crop_size,
        gap=args.gap,
    )
    print("[test] completed with official Ultralytics split_test()")


def main() -> None:
    args = parse_args()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    recommendation = resolve_split_settings(args)

    print(
        "[recommendation] "
        f"profile={recommendation['profile']} "
        f"median={recommendation['median']:.1f}px "
        f"q75={recommendation['q75']:.1f}px "
        f"q90={recommendation['q90']:.1f}px "
        f"gap={args.gap} "
        f"background_ratio={args.background_ratio:.3f} "
        f"max_background_per_image={args.max_background_per_image}"
    )
    print(f"[recommendation] reason={recommendation['reason']}")
    if recommendation["auto_enabled_background"]:
        print("[recommendation] auto-enabled sampled background retention because background_ratio > 0.")
    elif not args.allow_empty:
        print("[recommendation] --allow-empty is disabled, so empty tiles will be dropped completely.")

    for split in args.splits:
        if split == "test":
            split_test_split(args)
        else:
            split_labeled_split(args, split, rng)

    output_yaml = write_split_data_yaml(args.save_dir, args.source_yaml, args.splits)
    print(f"Saved sliced dataset to: {args.save_dir}")
    print(f"Generated dataset yaml: {output_yaml}")


if __name__ == "__main__":
    main()
