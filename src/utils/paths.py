from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    project_root: Path
    exp: str
    ver: str

    @property
    def exp_root(self) -> Path:
        return self.project_root / "experiments" / self.exp / self.ver

    @property
    def frames_dir(self) -> Path:
        return self.exp_root / "frames"

    @property
    def frames_candidates_dir(self) -> Path:
        return self.frames_dir / "_candidates"

    @property
    def colmap_dir(self) -> Path:
        return self.exp_root / "colmap"

    @property
    def colmap_db(self) -> Path:
        return self.colmap_dir / "database.db"

    @property
    def colmap_sparse0(self) -> Path:
        return self.colmap_dir / "sparse" / "0"

    @property
    def ns_data_dir(self) -> Path:
        return self.exp_root / "ns_data"

    @property
    def outputs_dir(self) -> Path:
        return self.exp_root / "outputs"

    @property
    def exports_dir(self) -> Path:
        return self.exp_root / "exports"

    @property
    def run_logs_dir(self) -> Path:
        return self.exp_root / "logs"
