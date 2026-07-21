#!/usr/bin/env python3
"""Export the FastAPI OpenAPI schema deterministically."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "openapi.json"
sys.path.insert(0, str(ROOT))


def render_schema() -> str:
    """Return the current OpenAPI schema as stable JSON."""
    from app.main import app

    return (
        json.dumps(
            app.openapi(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export or verify the X-AnyLabeling-Server OpenAPI schema."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path (default: docs/openapi.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the exported schema differs from the output file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    rendered = render_schema()

    if args.check:
        if (
            not output.exists()
            or output.read_text(encoding="utf-8") != rendered
        ):
            print(
                f"OpenAPI schema is stale: run {Path(__file__).name}",
                file=sys.stderr,
            )
            return 1
        print(f"OpenAPI schema is current: {output}")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote OpenAPI schema: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
