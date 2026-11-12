"""Run the abjad experiment on Modal GPUs.

Usage:
modal run modal_app.py::main                    # train all three models, then evaluate
modal run modal_app.py::main --max-steps 200    # quick end-to-end check
modal run modal_app.py::control                 # shallow-baseline control, run after main
"""

import io
import json
import sys
import urllib.request
from pathlib import Path

import modal

import evaluate
import train

APP_NAME = "abjad-transformer"
VOLUME_NAME = "abjad-experiment"
GPU_TYPE = "A10G"
TORCH_VERSION = "2.10.0"
TIMEOUT_SECONDS = 3600
DEFAULT_MAX_STEPS = 5000
BYTES_PER_MEGABYTE = 1e6

VOLUME_PATH = "/vol"
DATA_PATH = f"{VOLUME_PATH}/data/input.txt"
CHECKPOINT_DIR = f"{VOLUME_PATH}/checkpoints"
CONTROL_CHECKPOINT_DIR = f"{VOLUME_PATH}/checkpoints-control"
DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# The abjad system splits the baseline's depth between its two models so total size matches.
LAYERS_PER_TASK = {"baseline": 6, "skeleton": 4, "vocalizer": 2}
# The control baseline is as shallow as the skeleton model, to separate the effect of
# dropping vowels from the effect of simply running a smaller model.
CONTROL_BASELINE_LAYERS = LAYERS_PER_TASK["skeleton"]

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"torch=={TORCH_VERSION}")
    .add_local_python_source("abjad", "data", "model", "train", "evaluate")
)


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=TIMEOUT_SECONDS)
def download_dataset() -> int:
    """Fetch the dataset into the volume once. Returns its size in bytes."""
    path = Path(DATA_PATH)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(DATASET_URL, path)  # noqa: S310 - fixed https constant, not user input
        except OSError as error:
            message = f"Could not download dataset from {DATASET_URL}: {error}"
            raise RuntimeError(message) from error
        volume.commit()
    return path.stat().st_size


@app.function(image=image, gpu=GPU_TYPE, volumes={VOLUME_PATH: volume}, timeout=TIMEOUT_SECONDS)
def train_task(task: str, n_layer: int, out_dir: str, max_steps: int) -> float:
    """Train one model and store its best checkpoint in the volume."""
    args = train.parse_args(
        [
            *("--task", task),
            *("--data", DATA_PATH),
            *("--out_dir", out_dir),
            *("--n_layer", str(n_layer)),
            *("--max_steps", str(max_steps)),
        ]
    )
    best_val_loss = train.train(args)
    volume.commit()
    return best_val_loss


@app.function(image=image, gpu=GPU_TYPE, volumes={VOLUME_PATH: volume}, timeout=TIMEOUT_SECONDS)
def evaluate_models(baseline_dir: str = CHECKPOINT_DIR) -> evaluate.Report:
    args = evaluate.parse_args(
        ["--data", DATA_PATH, "--checkpoint_dir", CHECKPOINT_DIR, "--baseline_dir", baseline_dir]
    )
    return evaluate.evaluate(args)


def configure_console() -> None:
    # Windows consoles default to a legacy code page that cannot print every dataset character.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def save_report(results: evaluate.Report, filename: str) -> None:
    evaluate.print_report(results)
    Path(filename).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {filename}")


@app.local_entrypoint()
def main(max_steps: int = DEFAULT_MAX_STEPS) -> None:
    configure_console()
    dataset_bytes = download_dataset.remote()
    print(f"Dataset ready: {dataset_bytes / BYTES_PER_MEGABYTE:.1f} MB")

    jobs = [(task, n_layer, CHECKPOINT_DIR, max_steps) for task, n_layer in LAYERS_PER_TASK.items()]
    best_losses = train_task.starmap(jobs)
    for task, best_loss in zip(LAYERS_PER_TASK, best_losses, strict=True):
        print(f"{task}: best validation loss {best_loss:.4f}")

    save_report(evaluate_models.remote(), "results.json")


@app.local_entrypoint()
def control(max_steps: int = DEFAULT_MAX_STEPS) -> None:
    """Score a baseline as shallow as the skeleton model against the already-trained abjad system."""
    configure_console()
    best_loss = train_task.remote("baseline", CONTROL_BASELINE_LAYERS, CONTROL_CHECKPOINT_DIR, max_steps)
    print(f"control baseline ({CONTROL_BASELINE_LAYERS} layers): best validation loss {best_loss:.4f}")
    save_report(evaluate_models.remote(CONTROL_CHECKPOINT_DIR), "results_control.json")
