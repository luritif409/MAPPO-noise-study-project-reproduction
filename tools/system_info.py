#!/usr/bin/env python3
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/metadata/system_info.json")
    args = parser.parse_args()
    packages = {}
    for name in ["torch", "numpy", "gym", "pandas", "matplotlib", "tensorboardX"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        import torch
        cuda = {
            "available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        cuda = {"available": False, "error": str(exc)}
    result = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "cuda": cuda,
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(path.resolve())


if __name__ == "__main__":
    main()

