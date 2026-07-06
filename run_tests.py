


import sys
import subprocess
from pathlib import Path


def main():
    project_root = Path(__file__).parent
    print("=" * 70)
    print("  Biblio-HUB Unified Test Runner")
    print("=" * 70)
    print(f"  Project root: {project_root}")
    print("=" * 70)

    args = sys.argv[1:]
    if not args:
        print("\nNo test selector provided. Running the default regression suite...")
        args = ["-m", "not slow"]

    print(f"\nRunning command: pytest {' '.join(args)}\n")

    result = subprocess.run(
        [sys.executable, "-m", "pytest"] + args,
        cwd=str(project_root),
    )

    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("  All tests passed")
    else:
        print(f"  Tests failed (exit code: {result.returncode})")
    print("=" * 70)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
