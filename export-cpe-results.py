
import json
import sys
from pathlib import Path

import torch
import numpy as np

import __main__

class TrainConfig:
    pass

class BindingConfig:
    pass

class WorkspaceConfig:
    pass

__main__.TrainConfig = TrainConfig # type: ignore
__main__.BindingConfig = BindingConfig # type: ignore
__main__.WorkspaceConfig = WorkspaceConfig # type: ignore

KEYWORDS = (
    "history", "eval", "metric", "result", "attention",
    "probe", "jacobian", "activation", "config", "summary"
)

def clean(value):
    if isinstance(value, torch.Tensor):
        if value.numel() > 10000:
            return {
                "type": "tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype)
            }
        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):
        if value.size > 10000:
            return {"type": "array", "shape": list(value.shape)}
        return value.tolist()

    if isinstance(value, (np.integer, np.floating)):
        return value.item()

    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()
                if not str(k).lower().endswith("state_dict")}

    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(type(value).__name__)

def select_records(obj, path=""):
    found = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{path}.{key}" if path else str(key)

            if any(word in str(key).lower() for word in KEYWORDS):
                found[name] = clean(value)

            found.update(select_records(value, name))

    return found

def main():
    if len(sys.argv) < 2:
        print("Usage: python export_cpe_results.py PATH_TO_RUN_FOLDER")
        return

    run_dir = Path(sys.argv[1]).expanduser().resolve()
    output_dir = Path.home() / "Desktop" / "cpe_checkpoint_export"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {
        "world_model": run_dir / "checkpoint.pt",
        "binding_v0_1": run_dir / "binding_v0_1" / "binding_checkpoint.pt",
        "binding_v0_2": run_dir / "binding_v0_2" / "workspace_checkpoint.pt",
        "binding_emergent": run_dir / "binding_emergent" / "emergent_checkpoint.pt",
    }

    report = {}

    for name, path in checkpoints.items():
        if not path.exists():
            report[name] = {"error": f"Not found: {path}"}
            continue

        print(f"Reading {name}...")

        # Use only for your own trusted experiment checkpoints.
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )

        report[name] = {
            "file": path.name,
            "top_level_keys": list(checkpoint.keys())
                if isinstance(checkpoint, dict) else [],
            "records": select_records(checkpoint)
        }

    destination = output_dir / "cpe_results.json"
    destination.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8"
    )

    print(f"\nExport complete: {destination}")

if __name__ == "__main__":
    main()
