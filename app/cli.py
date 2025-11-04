"""Command-line interface for X-AnyLabeling Server."""

import argparse
import uvicorn

from app import __version__
from app.core.config import get_settings


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        prog='x-anylabeling-server',
        description='X-AnyLabeling Server - AI Model Inference Service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    )

    parser.add_argument(
        '--host',
        type=str,
        help='Server host (default: from config)',
    )

    parser.add_argument(
        '--port',
        type=int,
        help='Server port (default: from config)',
    )

    parser.add_argument(
        '--workers',
        type=int,
        help='Number of workers (default: from config)',
    )

    parser.add_argument(
        '--reload',
        action='store_true',
        help='Enable auto-reload (for development)',
    )

    args = parser.parse_args()

    # Load settings from config file
    settings = get_settings()

    # Override with command-line arguments if provided
    host = args.host or settings.server.host
    port = args.port or settings.server.port
    workers = args.workers or settings.server.workers
    reload = args.reload

    print(f"Starting X-AnyLabeling Server v{__version__}")
    print(f"Server: http://{host}:{port}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


if __name__ == "__main__":
    main()
