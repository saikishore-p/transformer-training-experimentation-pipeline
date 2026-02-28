from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pipeline.config import load_config
from pipeline.runner import run_pipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single transformer training experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to registry.json (default: <output_dir>/registry.json from config).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    config = load_config(config_path)
    result = run_pipeline(config, registry_path=args.registry)

    if result.get("regression_detected"):
        sys.exit(2)  # non-zero for regression detected


if __name__ == "__main__":
    main()
