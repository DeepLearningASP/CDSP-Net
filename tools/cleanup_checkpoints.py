"""Keep the best checkpoint from a completed training log.

Preview by default; use --apply to delete other epoch checkpoints.
"""

import argparse
import math
import os
import re
import stat
from pathlib import Path
from typing import List, Tuple


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_BEST_RESULT = re.compile(
    r"(?:\[[^\]\r\n]*\]\s*)?Best\s*Validation\s+Loss:\s*"
    + rf"(?P<loss>{_NUMBER})\s+at\s+epoch\s+(?P<epoch>[1-9]\d*)",
    re.IGNORECASE,
)
_CHECKPOINT = re.compile(r"Model_epoch([1-9]\d*)\.pth")


def _is_link(path: Path) -> bool:
    """Also detect Windows junctions/reparse points without following them."""
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def read_best_result(log_path: Path) -> Tuple[int, float]:
    """Read the last nonempty line, including an optional logger timestamp."""
    last_line = ""
    with log_path.open("r", encoding="utf-8-sig") as log_file:
        for line in log_file:
            if line.strip():
                last_line = line.strip()

    match = _BEST_RESULT.fullmatch(last_line)
    if match is None:
        raise ValueError("last nonempty log line is not a final Best Validation Loss result")

    loss = float(match.group("loss"))
    if not math.isfinite(loss):
        raise ValueError("best validation loss is not finite")
    return int(match.group("epoch")), loss


def find_logs(path: Path, recursive: bool) -> List[Path]:
    """Discover logs; never traverse directory links or Windows junctions."""
    if _is_link(path):
        raise ValueError(f"input path is a link or reparse point: {path}")
    if path.is_file():
        if path.name != "log.txt":
            raise ValueError("input file must be named log.txt")
        return [path]
    if not path.is_dir():
        raise ValueError(f"not a directory or log.txt: {path}")
    if not recursive:
        log_path = path / "log.txt"
        if not log_path.is_file():
            raise ValueError("no log.txt in this directory; use --recursive for a parent directory")
        return [log_path]

    logs = []

    def on_walk_error(error: OSError) -> None:
        raise error

    for directory, subdirectories, filenames in os.walk(
        path, followlinks=False, onerror=on_walk_error
    ):
        subdirectories[:] = sorted(
            name for name in subdirectories if not _is_link(Path(directory) / name)
        )
        if "log.txt" in filenames:
            logs.append(Path(directory) / "log.txt")
    return sorted(logs)


def cleanup_experiment(log_path: Path, apply: bool = False) -> Tuple[int, int]:
    """Return the number and total bytes of checkpoints selected for deletion."""
    if _is_link(log_path) or _is_link(log_path.parent):
        raise ValueError("log or experiment directory is a link or reparse point")
    log_signature = log_path.stat()
    epoch, loss = read_best_result(log_path)
    best_path = log_path.parent / f"Model_epoch{epoch}.pth"
    if not best_path.is_file() or _is_link(best_path):
        raise ValueError(f"best checkpoint is missing or linked: {best_path.name}")
    if best_path.stat().st_size == 0:
        raise ValueError(f"best checkpoint is empty: {best_path.name}")

    candidates = []
    for checkpoint in log_path.parent.iterdir():
        if _CHECKPOINT.fullmatch(checkpoint.name):
            if _is_link(checkpoint) or not checkpoint.is_file():
                raise ValueError(f"checkpoint is linked or not a regular file: {checkpoint.name}")
            if checkpoint != best_path:
                candidates.append(checkpoint)
    candidates.sort(key=lambda item: int(_CHECKPOINT.fullmatch(item.name).group(1)))
    sizes = [checkpoint.stat().st_size for checkpoint in candidates]

    print(f"\n[EXPERIMENT] {log_path.parent}")
    print(f"  KEEP {best_path.name} (validation loss={loss:g}, epoch={epoch})")

    # Reject logs changed by resumed training.
    current_signature = log_path.stat()
    if (current_signature.st_mtime_ns, current_signature.st_size) != (
        log_signature.st_mtime_ns, log_signature.st_size
    ):
        raise ValueError("log changed while planning; wait until training has finished")

    for checkpoint in candidates:
        if apply:
            checkpoint.unlink()
    return len(candidates), sum(sizes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Experiment directory, log.txt, or parent directory")
    parser.add_argument("--recursive", action="store_true", help="Find log.txt files recursively")
    parser.add_argument("--apply", action="store_true", help="Permanently delete unselected checkpoints")
    args = parser.parse_args()
    path = Path(os.path.abspath(args.path.expanduser()))
    try:
        logs = find_logs(path, args.recursive)
    except (OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    if not logs:
        parser.exit(1, f"ERROR: no log.txt found under {path}\n")

    print("Mode: APPLY (permanent deletion)" if args.apply else "Mode: PREVIEW (no files deleted)")
    skipped = 0
    for log_path in logs:
        try:
            cleanup_experiment(log_path, apply=args.apply)
        except (OSError, ValueError) as error:
            skipped += 1
            print(f"\n[SKIP/ERROR] {log_path.parent}: {error}")
            continue
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
