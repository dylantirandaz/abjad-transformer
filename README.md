# abjad-transformer

A small experiment: can a language model write the way Hebrew and Arabic scripts do, with the vowels left out?

Abjad scripts write mostly consonants and trust the reader to supply the vowels from context. This project
applies that to a character-level transformer:

1. A **skeleton model** generates text one character at a time, but only the non-vowel characters.
   `Once upon a time` becomes `nc pn  tm`.
2. A **vocalizer** reads the whole skeleton at once and fills every vowel back in, in a single pass.

The skeleton is about 70% of the original length, so the slow one-character-at-a-time model takes about 30%
fewer steps. The question is how much quality that costs.

## Fairness

Splitting text into a skeleton plus the removed vowels loses nothing, so

    P(text) = P(skeleton) x P(vowels | skeleton)

The two models together are therefore a proper language model, and their combined **bits per character** can
be compared directly with a normal character-level model of the same total size (the baseline).

## Layout

| File | Purpose |
| --- | --- |
| `abjad.py` | Lossless split of text into skeleton and vowel runs, and the inverse |
| `data.py` | Loads a text file and encodes it for each of the three models |
| `model.py` | One small transformer, causal or bidirectional |
| `train.py` | Trains the baseline, the skeleton model or the vocalizer |
| `evaluate.py` | Bits per character, vocalizer accuracy, generation speed, samples |
| `modal_app.py` | Runs the whole experiment on Modal GPUs |

## Running it

Install the dependencies:

    pip install -r requirements.txt

On Modal (trains the three models in parallel, then evaluates):

    modal run modal_app.py

Locally, with any UTF-8 text file at `data/input.txt`:

    python train.py --task baseline
    python train.py --task skeleton --n_layer 4
    python train.py --task vocalizer --n_layer 2
    python evaluate.py

## Checks

    pip install -r requirements-dev.txt
    ruff format --check .
    ruff check .
    mypy .
    pytest

Ruff runs with every rule enabled; the few exceptions are listed with reasons in `pyproject.toml`.
Mypy runs in strict mode.

## Status

The code and checks are complete. Results from the first full training run have not been recorded here yet.
