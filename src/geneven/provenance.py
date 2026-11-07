"""Run manifests: the five things needed to regenerate any number.

Every pipeline run writes one manifest recording the data snapshot, the pinned
reference-database versions, the code commit, the specification hash and the
random seed. A result without its manifest is not a Geneven result.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or "not-a-git-checkout"
    except Exception:
        return "unavailable"


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file, for pinning reference databases and input data."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class ReferenceVersions:
    """Pinned versions of every external knowledge source.

    These are recorded, never auto-updated. A phenotype assignment made under
    one CPIC release is not the same result as one made under the next.
    """

    pharmvar: str = "unset"
    cpic: str = "unset"
    pharmgkb: str = "unset"
    gnomad: str = "unset"
    reference_panel: str = "unset"
    genome_build: str = "GRCh38"

    def unset_fields(self) -> list[str]:
        return [k for k, v in asdict(self).items() if v == "unset"]


@dataclass
class RunManifest:
    """Written beside every set of outputs."""

    study_id: str
    spec_hash: str
    data_snapshot_id: str
    data_extracted_at: str
    references: ReferenceVersions
    random_seed: int
    code_commit: str = field(default_factory=_git_commit)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    run_started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    def record_input(self, name: str, path: str | Path) -> None:
        self.inputs[name] = file_digest(path)

    def record_output(self, name: str, path: str | Path) -> None:
        self.outputs[name] = file_digest(path)

    def warnings(self) -> list[str]:
        """Anything that would make this run non-reproducible."""
        issues = []
        unset = self.references.unset_fields()
        if unset:
            issues.append(f"reference versions not pinned: {', '.join(unset)}")
        if self.code_commit in {"unavailable", "not-a-git-checkout"}:
            issues.append("code commit unavailable: results are not traceable to a version")
        if self.data_snapshot_id in {"", "unset"}:
            issues.append("data snapshot identifier not set")
        return issues

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["warnings"] = self.warnings()
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        return path
