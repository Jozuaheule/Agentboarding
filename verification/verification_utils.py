from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class VerificationResult:
    test_id: str
    name: str
    status: str
    checks: int
    details: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


def pass_result(test_id: str, name: str, details: List[str]) -> dict:
    return VerificationResult(
        test_id=test_id,
        name=name,
        status="PASS",
        checks=len(details),
        details=details,
    ).to_dict()


def fail_result(test_id: str, name: str, details: List[str]) -> dict:
    return VerificationResult(
        test_id=test_id,
        name=name,
        status="FAIL",
        checks=len(details),
        details=details,
    ).to_dict()


def pass_result_with_meta(
    test_id: str,
    name: str,
    details: List[str],
    metadata: Optional[dict] = None,
) -> dict:
    result = pass_result(test_id, name, details)
    if metadata:
        result.update(metadata)
    return result


def fail_result_with_meta(
    test_id: str,
    name: str,
    details: List[str],
    metadata: Optional[dict] = None,
) -> dict:
    result = fail_result(test_id, name, details)
    if metadata:
        result.update(metadata)
    return result


def save_results_json(results: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
