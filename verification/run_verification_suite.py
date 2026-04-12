from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

SELF_DIR = Path(__file__).resolve().parent
if str(SELF_DIR) not in sys.path:
    sys.path.insert(0, str(SELF_DIR))

try:
    from verification.verification_utils import save_results_json
    from verification.verify_completion import run_completion_verification
    from verification.verify_environment import run_environment_verification
    from verification.verify_head_on import run_head_on_verification
    from verification.verify_initialization import run_initialization_verification
    from verification.verify_luggage import run_luggage_verification
    from verification.verify_movement import run_movement_verification
    from verification.verify_row_access import run_row_access_verification
    from verification.verify_spawning import run_spawning_verification
except ModuleNotFoundError:
    from verification_utils import save_results_json
    from verify_completion import run_completion_verification
    from verify_environment import run_environment_verification
    from verify_head_on import run_head_on_verification
    from verify_initialization import run_initialization_verification
    from verify_luggage import run_luggage_verification
    from verify_movement import run_movement_verification
    from verify_row_access import run_row_access_verification
    from verify_spawning import run_spawning_verification


def main() -> int:
    suites = [
        run_environment_verification,
        run_initialization_verification,
        run_spawning_verification,
        run_movement_verification,
        run_luggage_verification,
        run_row_access_verification,
        run_head_on_verification,
        run_completion_verification,
    ]

    all_results: list[dict] = []
    for suite in suites:
        all_results.extend(suite())

    passed = sum(r["status"] == "PASS" for r in all_results)
    total = len(all_results)

    print("\n" + "=" * 72)
    print("VERIFICATION SUITE SUMMARY")
    print("=" * 72)
    for result in all_results:
        print(f"[{result['status']}] {result['test_id']}: {result['name']}")
        for detail in result["details"]:
            print(f"    - {detail}")
    print("-" * 72)
    print(f"Total: {passed}/{total} passed")

    output_dir = Path(__file__).resolve().parent / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"verification_results_{timestamp}.json"
    save_results_json(all_results, output_file)
    print(f"Saved JSON report to: {output_file}")
    print("=" * 72 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
