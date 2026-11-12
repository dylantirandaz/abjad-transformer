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

    modal run modal_app.py::main

Then the control, a baseline as shallow as the skeleton model, scored against the same abjad models:

    modal run modal_app.py::control

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

## Results

One run, one seed. TinyStories (the 22 MB `TinyStoriesV2-GPT4-valid.txt` file, split 90/10), 5,000 training
steps per model, width 384, context 256, scored on 2.25 million held-out characters.

| System | Parameters | Bits per character | Characters per second |
| --- | --- | --- | --- |
| Baseline, 6 layers | 10.81M | 0.8436 | 424 |
| Baseline, 4 layers (control) | 7.26M | 0.8715 | 573 |
| Abjad: 4-layer skeleton model + 2-layer vocalizer | 10.96M | 0.8769 | 813 |

The abjad total is 0.8222 bits for the skeleton plus 0.0548 for the vowels.

What this shows:

- **Vowels are nearly free.** They are 28% of the characters but 6.2% of the bits. Given a true skeleton, the
  vocalizer restores 98.2% of vowel slots and 94.9% of whole words exactly, in one parallel pass.
- **Dropping vowels buys the full step saving at almost no quality cost.** Against the 4-layer baseline, which
  has the same depth as the skeleton model, the abjad system is 0.6% worse in bits per character and 1.42x
  faster on the same GPU. The skeleton is 72% of the text length, which predicts 1.39x.
- **At equal total size it is a trade, not a win.** Against the 6-layer baseline the abjad system is 3.9%
  worse and 1.95x faster. Most of that gap comes from the skeleton model being shallower, not from the
  missing vowels.

What it does not show:

- Whether this beats simply using a smaller model. The abjad system is faster than the 4-layer baseline but
  carries 51% more parameters, and no baseline was trained at matched speed.
- Whether the gaps are real. There is no measure of run-to-run noise, and 0.6% is small.
- Converged behaviour. Every model was still improving at step 5,000.
- Production speed. Generation was timed one sequence at a time with no key-value cache.
- Anything about subword tokenizers. All three systems are character-level.

Example of the vocalizer reading a skeleton from held-out text:

    skeleton:  Sh s nt hppy. Sh sys, "Wht r y dng? Ths rcrd ws  gft frm yr grndm.
    rebuilt:   She is not happy. She says, "What are you doing? Thes record was a gift from your grandma.
    original:  She is not happy. She says, "What are you doing? This record was a gift from your grandma.
