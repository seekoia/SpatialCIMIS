from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class PipelineConfig:
    base_path: Path
    file_identifier: str
    netcdf_variable: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    components: int = 3
    output_dir: Optional[Path] = None
    clip_path: Optional[Path] = None
    file_pattern: Optional[str] = None
    crs_epsg: Optional[int] = None


def parse_config_file(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    values: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()

    required = ["base_path", "file_identifier", "netcdf_variable"]
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    def _maybe_int(key: str) -> Optional[int]:
        raw = values.get(key)
        return int(raw) if raw else None

    output_dir = values.get("output_dir")
    clip_path = values.get("clip_path")
    file_pattern = values.get("file_pattern")
    crs_epsg = values.get("crs_epsg")

    return PipelineConfig(
        base_path=Path(values["base_path"]),
        file_identifier=values["file_identifier"],
        netcdf_variable=values["netcdf_variable"],
        start_year=_maybe_int("start_year"),
        end_year=_maybe_int("end_year"),
        components=int(values.get("components", 3)),
        output_dir=Path(output_dir) if output_dir else None,
        clip_path=Path(clip_path) if clip_path else None,
        file_pattern=file_pattern,
        crs_epsg=int(crs_epsg) if crs_epsg else None,
    )

