#!/usr/bin/env python3
"""Script to start the ECG analysis API server."""

import uvicorn
import sys
import os
from pathlib import Path
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.config import settings


def main():
    """Start the API server."""
    parser = argparse.ArgumentParser(description="Start ECG Analysis API Server")
    parser.add_argument("--host", default=settings.api_host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.api_port, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", default=settings.api_reload, 
                       help="Enable auto-reload")
    parser.add_argument("--log-level", default=settings.api_log_level, 
                       choices=["critical", "error", "warning", "info", "debug", "trace"],
                       help="Log level")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    
    args = parser.parse_args()
    
    print(f"Starting ECG Analysis API Server...")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Log Level: {args.log_level}")
    print(f"Workers: {args.workers}")
    print(f"Project Root: {project_root}")
    
    # Change to project root directory
    os.chdir(project_root)
    
    # Start the server
    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        workers=args.workers if not args.reload else 1,  # Can't use workers with reload
        access_log=True
    )


if __name__ == "__main__":
    main()