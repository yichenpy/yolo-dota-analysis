# Analysis Scripts for YOLO OBB

This directory adds standalone analysis code for a full server-side YOLO OBB project without modifying the training entry points.

Generated scripts:

- `analysis/analyze_errors.py`
  - Reads validation GT and prediction results.
  - Optionally records an official Ultralytics `val()` pass for direct comparison with training logs.
  - Counts misses and false positives.
  - Summarizes misses and false positives by class.
  - Summarizes missed-target size distribution.
  - Saves bar charts and size-based miss-rate charts.

- `analysis/analyze_object_sizes.py`
  - Scans labeled dataset splits.
  - Computes OBB target-size statistics for the whole dataset.
  - Saves summary JSON, per-class CSV, histogram, and coarse bucket chart.

- `analysis/analyze_detection_layers.py`
  - Loads the YOLO OBB model weights.
  - Optionally records an official Ultralytics `val()` pass for direct comparison with training logs.
  - Uses forward hooks to inspect detection-layer and branch outputs.
  - Prints feature map shapes, layer strides, and candidate counts.
  - Optionally adds a heuristic size-to-layer and miss-rate summary.

- `analysis/common_obb.py`
  - Shared OBB geometry, dataset path resolution, label parsing, IoU, and JSON helpers.

## Dependency Installation

Install these in the full server environment where the complete project and data exist:

```bash
pip install ultralytics torch torchvision matplotlib pillow pyyaml numpy
```

If your server already has a working training environment, reuse that environment first.

## Official Metrics vs Custom Analysis

This distinction is important.

- Official comparable metrics:
  - Produced by Ultralytics `model.val()`
  - These are the metrics you should compare with `output.log`
  - Saved to:
    - `analysis/outputs/errors/official_val_metrics.json`
    - `analysis/outputs/detection_layers/official_val_metrics.json`

- Custom analysis metrics:
  - Produced by the standalone analysis scripts
  - Used for miss/false breakdown by class, size, and heuristic layer responsibility
  - These are diagnostic metrics, not replacements for Ultralytics official `P/R/mAP`
  - Saved to:
    - `analysis/outputs/errors/summary.json`
    - `analysis/outputs/detection_layers/layer_summary.json`

Use this rule:

- compare with training log: `official_val_metrics.json`
- analyze failure patterns: `summary.json` and `layer_summary.json`

## Paths You Need to Provide

At runtime, provide the following paths:

- `--dataset`
  - Example: `datasets/DOTA/data.yaml`
  - Or your split/tiled dataset YAML if analysis should run on the sliced dataset instead.

- `--predictions` for `analyze_errors.py`
  - A directory of per-image prediction `.txt` files, or
  - A `.json` prediction file.

- `--weights` for:
  - `analyze_errors.py` if you want the script to run inference directly instead of reading exported predictions.
  - `analyze_detection_layers.py` because it needs a live model forward pass.

- `--output-dir`
  - Optional. If omitted, outputs go under `analysis/outputs/...`.

## Expected Label and Prediction Formats

### GT label format

The scripts assume Ultralytics OBB labels:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

The coordinates can be normalized or absolute. The script will scale them to pixels when values look normalized.

### Prediction `.txt` format

Default assumption:

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4 conf
```

Alternative supported layout:

```text
class_id conf x1 y1 x2 y2 x3 y3 x4 y4
```

Select it with:

```bash
--prediction-layout class_conf_xyxyxyxy
```

### Prediction `.json` format

The JSON parser accepts common dictionary/list layouts with fields such as:

- `image` or `image_id` or `file_name`
- `class_id` or `category_id`
- `points` or `polygon` or `obb` or `segmentation`
- `confidence` or `score`

If your exported result format does not match, modify:

- `load_predictions_from_json()` in `analysis/analyze_errors.py`
- `load_predictions_from_text_dir()` in `analysis/analyze_errors.py`
- `parse_label_line()` in `analysis/common_obb.py`

## OBB Size Definition

The scripts are OBB-aware and do not use axis-aligned box size by default.

Supported size metrics:

- `area`
  - Polygon area in square pixels.

- `equivalent_side`
  - `sqrt(area)` in pixels.
  - Default because it is rotation-invariant and easier to interpret than raw area.

- `long_edge`
  - Average long side length of the rotated box polygon.

- `short_edge`
  - Average short side length of the rotated box polygon.

If you want a different definition later, modify:

- `polygon_metrics()` in `analysis/common_obb.py`

## Example Commands

### 1. Dataset-wide target-size analysis

Run this first to understand the dataset size distribution before error interpretation:

```bash
python analysis/analyze_object_sizes.py \
  --dataset datasets/DOTA/data.yaml \
  --splits train val \
  --size-metric equivalent_side \
  --binning quantile \
  --output-dir analysis/outputs/object_sizes
```

### 2. Error analysis using exported predictions

```bash
python analysis/analyze_errors.py \
  --dataset datasets/DOTA/data.yaml \
  --split val \
  --predictions path/to/predictions/labels \
  --prediction-format txt \
  --prediction-layout class_xyxyxyxy_conf \
  --size-metric equivalent_side \
  --save-details \
  --output-dir analysis/outputs/errors
```

### 3. Error analysis by directly running inference

Use this if you do not already have an exported prediction file:

```bash
python analysis/analyze_errors.py \
  --dataset datasets/DOTA/data.yaml \
  --split val \
  --weights path/to/best.pt \
  --imgsz 1024 \
  --batch 1 \
  --device 0 \
  --half \
  --max-det 100 \
  --size-metric equivalent_side \
  --save-details \
  --output-dir analysis/outputs/errors
```

By default, this also writes:

```text
analysis/outputs/errors/official_val_metrics.json
```

If you want to skip the extra official validation pass:

```bash
--skip-official-val
```

### 4. Detection-layer / branch analysis

```bash
python analysis/analyze_detection_layers.py \
  --dataset datasets/DOTA/data.yaml \
  --weights path/to/best.pt \
  --split val \
  --imgsz 1024 \
  --batch 1 \
  --device 0 \
  --half \
  --max-det 100 \
  --max-images 8 \
  --size-metric equivalent_side \
  --output-dir analysis/outputs/detection_layers
```

By default, this also writes:

```text
analysis/outputs/detection_layers/official_val_metrics.json
```

If you only want tensor shapes and feature map sizes without the heuristic miss-rate section:

```bash
python analysis/analyze_detection_layers.py \
  --dataset datasets/DOTA/data.yaml \
  --weights path/to/best.pt \
  --split val \
  --skip-performance \
  --output-dir analysis/outputs/detection_layers
```

## Output Locations

Default output locations:

- `analysis/outputs/object_sizes/`
- `analysis/outputs/errors/`
- `analysis/outputs/detection_layers/`

Typical outputs:

- `object_size_summary.json`
- `object_size_histogram.png`
- `summary.json`
- `official_val_metrics.json`
- `missed_by_class.png`
- `false_by_class.png`
- `missed_size_histogram.png`
- `miss_rate_by_size.png`
- `layer_summary.json`
- `candidate_predictions_by_layer.png`

## Notes About Detection-Layer Analysis

`analyze_detection_layers.py` has two levels of analysis:

- Direct:
  - detection-head input tensor shapes
  - branch output tensor shape signatures
  - feature map sizes
  - stride estimates
  - candidate counts

- Heuristic:
  - assigns GT objects to layers according to target size vs layer stride
  - summarizes miss rate by inferred responsible layer

This heuristic is useful for deciding whether a larger feature map might be worth testing, but it is not a strict proof of branch responsibility.

`layer_summary.json` should therefore be interpreted as:

- official validation numbers: from `official_val_metrics.json`
- architecture diagnosis and scale trends: from `layer_summary.json`

## When Result Formats or Model Internals Differ

Most likely adjustment points:

- Prediction parsing:
  - `load_predictions_from_text_dir()` in `analysis/analyze_errors.py`
  - `load_predictions_from_json()` in `analysis/analyze_errors.py`

- Ultralytics OBB result extraction:
  - `load_predictions_from_model()` in `analysis/analyze_errors.py`
  - `run_predictions()` in `analysis/analyze_detection_layers.py`

- Detection head discovery / hook targets:
  - `find_detection_head()` in `analysis/analyze_detection_layers.py`
  - `register_hooks()` in `analysis/analyze_detection_layers.py`

These are the first places to edit if the full server project uses a different Ultralytics version or a custom OBB head implementation.
