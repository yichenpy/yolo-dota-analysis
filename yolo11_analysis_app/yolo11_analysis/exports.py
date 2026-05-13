from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_directory, timestamp_string, to_serializable, write_json


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    return dataframe.to_csv(index=False).encode("utf-8-sig")


def json_to_bytes(payload: Any) -> bytes:
    import json

    return json.dumps(to_serializable(payload), ensure_ascii=False, indent=2).encode("utf-8")


def save_analysis_run(
    output_root: str | Path,
    run_name: str,
    payload: dict[str, Any],
    tables: dict[str, pd.DataFrame] | None = None,
    figures: dict[str, Any] | None = None,
) -> Path:
    run_dir = ensure_directory(Path(output_root) / f"{timestamp_string()}_{run_name}")
    write_json(run_dir / "summary.json", payload)

    for table_name, dataframe in (tables or {}).items():
        if dataframe is not None and not dataframe.empty:
            dataframe.to_csv(run_dir / f"{table_name}.csv", index=False, encoding="utf-8-sig")

    for figure_name, figure in (figures or {}).items():
        if figure is not None:
            figure.savefig(run_dir / f"{figure_name}.png", dpi=200, bbox_inches="tight")

    return run_dir
