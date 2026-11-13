"""Run the chord-token experiment on Modal.

Usage:
modal run modal_chords.py::count             # passes per character for BPE and for chords, CPU only
modal run modal_chords.py::count --text c4   # the same count on English web text (one file of C4)
modal run modal_chords.py::prepare           # fetch training text, tokenize, build chord streams
modal run --detach modal_chords.py::train    # train every arm in parallel, one GPU each
modal run modal_chords.py::evaluate          # score every trained arm
modal run modal_chords.py::equal_text        # controls: a chord arm on equal text, and a baseline at its step count
modal run modal_chords.py::capacity          # controls: much smaller models, trained until they stop improving
modal run modal_chords.py::evaluate --run equal_text
modal run modal_chords.py::evaluate --run small
modal run --detach modal_chords.py::prepare_c4   # web text: fetch C4 files, tokenize, build chord streams
modal run --detach modal_chords.py::train_c4     # web text: train a BPE baseline and each chord setting
modal run modal_chords.py::evaluate --run c4
"""

import gzip
import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path

import modal

import chord_count
import chord_evaluate
import chord_prepare
import chord_train
from bpe import END_OF_TEXT
from chord_model import HOST_FIRST, INDEPENDENT, RUN_FIRST
from console import configure_console

APP_NAME = "chord-tokens"
VOLUME_NAME = "abjad-experiment"
GPU_TYPE = "A10G"
TORCH_VERSION = "2.10.0"
NUMPY_VERSION = "2.3.5"
TOKENIZERS_VERSION = "0.22.2"
TIMEOUT_SECONDS = 3600
CPU_COUNT = 8
MEMORY_MB = 32768
DEFAULT_MAX_STEPS = 5000

VOLUME_PATH = "/vol"
VALIDATION_TEXT_PATH = f"{VOLUME_PATH}/data/input.txt"
TRAIN_TEXT_PATH = f"{VOLUME_PATH}/data/tinystories_train_head.txt"
TRAIN_TEXT_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
# Only the start of the 2.2 GB training file is fetched. That is enough text that no arm sees it more than twice.
TRAIN_TEXT_BYTES = 400 * 1024 * 1024
PREPARED_DIR = f"{VOLUME_PATH}/chords/prepared"
CHECKPOINT_DIR = f"{VOLUME_PATH}/chords/checkpoints"
EQUAL_TEXT_CHECKPOINT_DIR = f"{VOLUME_PATH}/chords/checkpoints-equal-text"
SMALL_CHECKPOINT_DIR = f"{VOLUME_PATH}/chords/checkpoints-small"
RESULTS_DIR = f"{VOLUME_PATH}/chords/results"
DONE_SUFFIX = ".done"

# English web text: files from the validation split of C4, which is cleaned Common Crawl text.
# File 0 is held out for scoring. Files 1 to 4 are the training text for the web-text models.
TINYSTORIES_TEXT = "tinystories"
C4_TEXT = "c4"
COUNT_TEXTS = (TINYSTORIES_TEXT, C4_TEXT)
C4_BASE_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en"
C4_HELD_OUT_FILES = (0,)
C4_TRAIN_FILES = (1, 2, 3, 4)
C4_TEXT_PATH = f"{VOLUME_PATH}/data/c4-validation-00000-of-00008.txt"
C4_TRAIN_TEXT_PATH = f"{VOLUME_PATH}/data/c4-validation-00001-to-00004-of-00008.txt"
C4_PREPARED_DIR = f"{VOLUME_PATH}/chords/prepared-c4"
C4_CHECKPOINT_DIR = f"{VOLUME_PATH}/chords/checkpoints-c4"
C4_BPE_VOCAB_SIZE = 32768
# BPE keeps improving with vocabulary size on web text, so chords built on a 32,768-token BPE are
# compared with BPE vocabularies up to four times larger.
C4_COUNT_SETTINGS = chord_count.CountSettings(
    bpe_vocab_sizes=(8192, 16384, C4_BPE_VOCAB_SIZE, 65536, 131072),
    chord_bpe_vocab_size=C4_BPE_VOCAB_SIZE,
    satellite_counts=(32, 64, 128, 256),
    run_counts=(4096, 16384),
)
# 4,096 runs keeps the chord models' extra lookup tables small. On web text, 16,384 runs would
# save at most 3.3% more passes.
C4_SETTINGS = ("64:4096", "128:4096", "256:4096")
C4_ARM_SETTINGS: tuple[str | None, ...] = (None, "k64_r4096", "k128_r4096", "k256_r4096")

# satellites:runs pairs, from cautious to aggressive.
SETTINGS = ("32:1024", "64:4096", "128:4096")
MAIN_SETTING = "k64_r4096"
# The capacity control: a model small enough to stop improving within the run, trained for
# longer, so that what limits it is its size and not its training time.
SMALL_MODEL_ARGUMENTS = ("--n_layer", "2", "--n_head", "4", "--n_embd", "128")
SMALL_MAX_STEPS = 10_000
SMALL_SETTINGS: tuple[str | None, ...] = (None, "k32_r1024", MAIN_SETTING, "k128_r4096")
# Each arm is (chord setting, or None for the BPE baseline; pick order). The main setting is
# trained with every pick order to measure how much the link between the two picks matters.
ARMS: tuple[tuple[str | None, str], ...] = (
    (None, RUN_FIRST),
    ("k32_r1024", RUN_FIRST),
    (MAIN_SETTING, RUN_FIRST),
    (MAIN_SETTING, HOST_FIRST),
    (MAIN_SETTING, INDEPENDENT),
    ("k128_r4096", RUN_FIRST),
)

# Each named run keeps its prepared data and its checkpoints here.
MAIN_RUN = "main"
EQUAL_TEXT_RUN = "equal_text"
SMALL_RUN = "small"
C4_RUN = "c4"
RUN_DIRS = {
    MAIN_RUN: (PREPARED_DIR, CHECKPOINT_DIR),
    EQUAL_TEXT_RUN: (PREPARED_DIR, EQUAL_TEXT_CHECKPOINT_DIR),
    SMALL_RUN: (PREPARED_DIR, SMALL_CHECKPOINT_DIR),
    C4_RUN: (C4_PREPARED_DIR, C4_CHECKPOINT_DIR),
}

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"torch=={TORCH_VERSION}")
    .pip_install(f"numpy=={NUMPY_VERSION}", f"tokenizers=={TOKENIZERS_VERSION}")
    .add_local_python_source(
        "abjad",
        "bpe",
        "chord_count",
        "chord_evaluate",
        "chord_model",
        "chord_prepare",
        "chord_train",
        "chords",
        "console",
        "data",
        "evaluate",
        "model",
        "train",
    )
)


def download_training_text() -> None:
    """Fetch the start of the training file into the volume once."""
    path = Path(TRAIN_TEXT_PATH)
    if path.exists():
        return
    byte_range = {"Range": f"bytes=0-{TRAIN_TEXT_BYTES - 1}"}
    request = urllib.request.Request(TRAIN_TEXT_URL, headers=byte_range)  # noqa: S310 - fixed https constant
    try:
        with urllib.request.urlopen(request) as response:  # noqa: S310 - fixed https constant
            head: bytes = response.read(TRAIN_TEXT_BYTES)
    except OSError as error:
        message = f"Could not download training text from {TRAIN_TEXT_URL}: {error}"
        raise RuntimeError(message) from error
    # Keep whole lines only, so no character is cut in half.
    path.write_bytes(head[: head.rfind(b"\n") + 1])


def c4_url(file_index: int) -> str:
    return f"{C4_BASE_URL}/c4-validation.{file_index:05d}-of-00008.json.gz"


def parse_c4_records(compressed: bytes) -> list[str]:
    """Return the text of every document in a gzipped C4 file."""
    # Split on newlines only: a JSON string can hold other line-separator characters.
    records = gzip.decompress(compressed).decode("utf-8").split("\n")
    texts: list[str] = [json.loads(record)["text"] for record in records if record]
    return texts


def fetch_c4_documents(url: str) -> list[str]:
    """Download one C4 file and return the text of every document in it."""
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https host
            compressed: bytes = response.read()
        texts = parse_c4_records(compressed)
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, KeyError) as error:
        message = f"Could not read C4 documents from {url}: {error}"
        raise RuntimeError(message) from error
    return texts


def download_c4_text(file_indices: tuple[int, ...], text_path: str) -> None:
    """Store C4 files in the volume once, as plain text with a marker after each document."""
    path = Path(text_path)
    if path.exists():
        return
    documents = [text for index in file_indices for text in fetch_c4_documents(c4_url(index))]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written under another name first, so an interrupted run never leaves a partial file that looks complete.
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text("".join(f"{text}\n{END_OF_TEXT}\n" for text in documents), encoding="utf-8")
    partial.replace(path)


def keep_in_volume(report: Mapping[str, object], name: str) -> None:
    """Save a report in the volume too, so it survives a caller that lost its connection."""
    path = Path(RESULTS_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    volume.commit()


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=CPU_COUNT, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS)
def count_passes(text: str) -> chord_count.CountReport:
    if text == C4_TEXT:
        download_c4_text(C4_HELD_OUT_FILES, C4_TEXT_PATH)
        volume.commit()
        report = chord_count.count(C4_TEXT_PATH, C4_COUNT_SETTINGS)
    else:
        report = chord_count.count(VALIDATION_TEXT_PATH)
    keep_in_volume(report, f"count_{text}.json")
    return report


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=CPU_COUNT, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS)
def prepare_data() -> chord_prepare.PreparedMeta:
    download_training_text()
    args = chord_prepare.parse_args(
        [
            *("--train_text", TRAIN_TEXT_PATH),
            *("--val_text", VALIDATION_TEXT_PATH),
            *("--out_dir", PREPARED_DIR),
            *("--settings", *SETTINGS),
        ]
    )
    meta = chord_prepare.prepare(args)
    volume.commit()
    return meta


@app.function(image=image, volumes={VOLUME_PATH: volume}, cpu=CPU_COUNT, memory=MEMORY_MB, timeout=TIMEOUT_SECONDS)
def prepare_c4_data() -> chord_prepare.PreparedMeta:
    download_c4_text(C4_HELD_OUT_FILES, C4_TEXT_PATH)
    download_c4_text(C4_TRAIN_FILES, C4_TRAIN_TEXT_PATH)
    volume.commit()
    args = chord_prepare.parse_args(
        [
            *("--train_text", C4_TRAIN_TEXT_PATH),
            *("--val_text", C4_TEXT_PATH),
            *("--out_dir", C4_PREPARED_DIR),
            *("--bpe_vocab_size", str(C4_BPE_VOCAB_SIZE)),
            *("--settings", *C4_SETTINGS),
        ]
    )
    meta = chord_prepare.prepare(args)
    volume.commit()
    return meta


@app.function(image=image, gpu=GPU_TYPE, volumes={VOLUME_PATH: volume}, timeout=TIMEOUT_SECONDS)
def train_one_arm(
    setting: str | None,
    order: str,
    max_steps: int,
    run: str = MAIN_RUN,
    model_arguments: tuple[str, ...] = (),
) -> float:
    """Train one arm of a named run and store its best checkpoint in the volume."""
    data_dir, out_dir = RUN_DIRS[run]
    setting_arguments = [] if setting is None else ["--setting", setting]
    args = chord_train.parse_args(
        [
            *("--data_dir", data_dir),
            *("--out_dir", out_dir),
            *("--order", order),
            *("--max_steps", str(max_steps)),
            *setting_arguments,
            *model_arguments,
        ]
    )
    # The marker lets a caller that lost its connection tell a finished arm from one still training.
    marker = Path(out_dir) / f"{chord_train.arm_name(setting, order)}{DONE_SUFFIX}"
    marker.unlink(missing_ok=True)
    best_val_loss = chord_train.train_arm(args)
    marker.write_text(f"{best_val_loss:.6f}\n", encoding="utf-8")
    volume.commit()
    return best_val_loss


@app.function(image=image, timeout=TIMEOUT_SECONDS)
def train_all_arms(max_steps: int) -> list[float]:
    """Train every arm in parallel, one GPU each.

    This coordinator runs on Modal so the arms belong to it. A detached run only keeps the
    last function the caller triggered alive, which would otherwise be a single arm.
    """
    return list(train_one_arm.starmap([(setting, order, max_steps) for setting, order in ARMS]))


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=TIMEOUT_SECONDS)
def train_equal_text_arms() -> tuple[int, list[float]]:
    """Train the short-run controls: the main chord arm and a baseline, for the same reduced number of steps.

    A chord covers more text than a token, so at equal steps the chord arms read more text.
    Here the chord arm stops once it has read what the baseline reads in a full run. The
    short baseline then shows whether the two still match when compute is equal at that level.
    Returns (steps trained, best validation losses for the chord arm and the baseline).
    """
    meta = chord_prepare.load_meta(PREPARED_DIR)
    text_share = meta["settings"][MAIN_SETTING]["train_chords"] / meta["train_tokens"]
    steps = round(DEFAULT_MAX_STEPS * text_share)
    arms = [(setting, RUN_FIRST, steps, EQUAL_TEXT_RUN) for setting in (MAIN_SETTING, None)]
    return steps, list(train_one_arm.starmap(arms))


@app.function(image=image, timeout=TIMEOUT_SECONDS)
def train_small_arms() -> list[float]:
    """Train the capacity controls: a baseline and every chord setting with a much smaller model.

    A small model stops improving within the run. The gap that remains between chords and
    the baseline at that point is the cost of chords when model size is the limit.
    """
    arms = [(setting, RUN_FIRST, SMALL_MAX_STEPS, SMALL_RUN, SMALL_MODEL_ARGUMENTS) for setting in SMALL_SETTINGS]
    return list(train_one_arm.starmap(arms))


@app.function(image=image, timeout=TIMEOUT_SECONDS)
def train_c4_arms() -> list[float]:
    """Train the web-text arms: a BPE baseline and each chord setting, with the main run's model and steps."""
    arms = [(setting, RUN_FIRST, DEFAULT_MAX_STEPS, C4_RUN) for setting in C4_ARM_SETTINGS]
    return list(train_one_arm.starmap(arms))


@app.function(image=image, gpu=GPU_TYPE, volumes={VOLUME_PATH: volume}, timeout=TIMEOUT_SECONDS)
def evaluate_arms(run: str) -> chord_evaluate.ChordEvalReport:
    data_dir, checkpoint_dir = RUN_DIRS[run]
    args = chord_evaluate.parse_args(["--data_dir", data_dir, "--checkpoint_dir", checkpoint_dir])
    report = chord_evaluate.evaluate(args)
    keep_in_volume(report, f"evaluate_{run}.json")
    return report


@app.local_entrypoint()
def count(text: str = TINYSTORIES_TEXT) -> None:
    configure_console()
    if text not in COUNT_TEXTS:
        message = f"Unknown text {text!r}. Choose one of: {', '.join(COUNT_TEXTS)}"
        raise ValueError(message)
    report = count_passes.remote(text)
    chord_count.print_report(report)
    results_file = "results_count.json" if text == TINYSTORIES_TEXT else f"results_count_{text}.json"
    Path(results_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {results_file}")


@app.local_entrypoint()
def prepare() -> None:
    configure_console()
    print(json.dumps(prepare_data.remote(), indent=2))


@app.local_entrypoint()
def prepare_c4() -> None:
    configure_console()
    print(json.dumps(prepare_c4_data.remote(), indent=2))


@app.local_entrypoint()
def train(max_steps: int = DEFAULT_MAX_STEPS) -> None:
    configure_console()
    best_losses = train_all_arms.remote(max_steps)
    for (setting, order), best_loss in zip(ARMS, best_losses, strict=True):
        print(f"{chord_train.arm_name(setting, order)}: best validation loss {best_loss:.4f}")


@app.local_entrypoint()
def train_c4() -> None:
    configure_console()
    best_losses = train_c4_arms.remote()
    for setting, best_loss in zip(C4_ARM_SETTINGS, best_losses, strict=True):
        print(f"c4 {chord_train.arm_name(setting, RUN_FIRST)}: best validation loss {best_loss:.4f}")


@app.local_entrypoint()
def equal_text() -> None:
    configure_console()
    steps, (chord_loss, baseline_loss) = train_equal_text_arms.remote()
    print(f"{MAIN_SETTING} trained for {steps} steps on equal text: best validation loss {chord_loss:.4f}")
    print(f"baseline trained for the same {steps} steps: best validation loss {baseline_loss:.4f}")


@app.local_entrypoint()
def capacity() -> None:
    configure_console()
    best_losses = train_small_arms.remote()
    for setting, best_loss in zip(SMALL_SETTINGS, best_losses, strict=True):
        print(f"small {chord_train.arm_name(setting, RUN_FIRST)}: best validation loss {best_loss:.4f}")


@app.local_entrypoint()
def evaluate(run: str = MAIN_RUN) -> None:
    configure_console()
    if run not in RUN_DIRS:
        message = f"Unknown run {run!r}. Choose one of: {', '.join(RUN_DIRS)}"
        raise ValueError(message)
    results_file = "results_chords.json" if run == MAIN_RUN else f"results_chords_{run}.json"
    report = evaluate_arms.remote(run)
    chord_evaluate.print_report(report)
    Path(results_file).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {results_file}")
