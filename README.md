# abjad-transformer

Two small experiments on one question: can a language model produce text in fewer passes than standard
tokenization needs, without losing quality?

- **Chord tokens** fold grammar tokens into the content token that follows them, so one pass emits both.
- **Abjad** drops the vowels, generates only the consonant skeleton, and fills the vowels back in afterwards.

Both are scored in **bits per character**, which does not depend on how the text is cut into tokens.

## Chord tokens

### The idea

A model that uses byte-pair encoding (BPE) spends one full pass on every token, and BPE never merges across
a space, so a token is at most one word. A large share of those passes go on a few very common tokens such
as `.` `,` ` the` ` and` ` a` ` to`: on TinyStories the 32 most frequent tokens are 46% of all tokens. A
chord makes each of them ride along with the next ordinary token.

For example, if `She`, ` to`, ` the`, ` with`, ` her` and `.` are among the frequent tokens:

    She went to the park with her dog.

    BPE:     She | went | to | the | park | with | her | dog | .          9 passes
    chords:  [She] went  ·  [to the] park  ·  [with her] dog  ·  [.] ...   3 passes and the start of a fourth

Merging those words into bigger tokens would need one vocabulary entry per combination. A chord instead
makes two picks from two short lists, so every combination is available without being stored.

### How it works

1. Train an ordinary BPE tokenizer.
2. Call its most frequent tokens **satellites**. Nothing else is needed to choose them: no word list and no
   knowledge of the language.
3. Group the token stream into **chords**. Each chord is a **run** of satellites plus the **host** token that
   follows. Turning chords back into tokens is concatenation, so nothing is lost.
4. Keep a fixed list of the most frequent runs. A run that is not on the list is spread over several chords,
   each taking the longest known run and promoting the next satellite to host.
5. In the model, each position reads the sum of a run embedding and a host embedding. To predict the next
   chord it makes one pick, looks that pick up in a table, adds the result to its hidden state, and makes
   the second pick. That table lookup is the only link between the two picks. There is no second network.

### Results

**Bottom line.** On TinyStories, chord tokens match BPE's quality with about half the passes. On English web
text the saving is smaller and not free: at equal model size, chords generate 1.5x to 1.6x faster than BPE but
are 2.6% to 4.3% worse in bits per character.

**Status.** Work on chord tokens stopped after the web-text test. At equal quality the estimated gain on web
text, about 1.2x to 1.3x faster generation, was too small to justify replacing BPE.

One run, one seed. TinyStories: the first 400 MB of `TinyStoriesV2-GPT4-train.txt` for training (419 million
characters, 102 million BPE tokens) and all of `TinyStoriesV2-GPT4-valid.txt` for scoring (22.5 million
characters). BPE vocabulary 8,192. Every model is the same transformer with context 256.

**Passes needed on TinyStories, before any training.** BPE stops improving once tokens reach whole words.
Chords keep going. Counted on a held-out tenth of the validation file.

| Tokenization | Characters per pass | Passes relative to BPE |
| --- | --- | --- |
| BPE, 4,096 tokens | 3.98 | |
| BPE, 8,192 tokens | 4.11 | 1.00 |
| BPE, 16,384 tokens | 4.13 | |
| Chords, 32 satellites, 1,024 runs | 7.53 | 0.55 |
| Chords, 64 satellites, 4,096 runs | 9.11 | 0.45 |
| Chords, 128 satellites, 4,096 runs | 10.45 | 0.39 |

**Passes needed on English web text, before any training.** One file of C4, which is cleaned Common Crawl
text (`en/c4-validation.00000-of-00008.json.gz`), counted on a held-out tenth: 9.6 million characters. On web
text BPE keeps improving as its vocabulary grows, so chords built on a 32,768-token BPE are also compared with
a BPE four times larger, about the size of tokenizers in current use.

| Tokenization | Characters per pass | Passes vs 32,768-token BPE | Passes vs 131,072-token BPE |
| --- | --- | --- | --- |
| BPE, 8,192 tokens | 3.84 | | |
| BPE, 16,384 tokens | 4.20 | | |
| BPE, 32,768 tokens | 4.47 | 1.00 | |
| BPE, 65,536 tokens | 4.65 | | |
| BPE, 131,072 tokens | 4.75 | | 1.00 |
| Chords, 32 satellites, 12,583 runs | 6.67 | 0.67 | 0.71 |
| Chords, 64 satellites, 16,384 runs | 7.28 | 0.61 | 0.65 |
| Chords, 128 satellites, 16,384 runs | 7.86 | 0.57 | 0.61 |
| Chords, 256 satellites, 16,384 runs | 8.40 | 0.53 | 0.57 |

With 32 satellites only 12,583 different runs occur. Limiting the list to 4,096 runs costs up to 3.3% more
passes. The saving is smaller than on TinyStories because web text has fewer grammar tokens: the 32 most
frequent tokens are 33% of all tokens here, against 46% on TinyStories. The most frequent runs are still
grammar: ` and`, ` the`, ` of the`, ` in the`.

**Quality and speed at equal training steps.** 6 layers, width 384, 5,000 steps each.

| Model | Parameters | Passes vs BPE | Bits per character | Quality vs BPE | Speed vs BPE |
| --- | --- | --- | --- | --- | --- |
| BPE | 17.0M | 1.00 | 0.5512 | | 1.00x |
| Chords, 32 satellites | 18.2M | 0.54 | 0.5474 | 0.7% better | 1.58x |
| Chords, 64 satellites | 21.7M | 0.45 | 0.5532 | 0.4% worse | 1.97x |
| Chords, 128 satellites | 21.7M | 0.39 | 0.5587 | 1.4% worse | 1.99x |

Speed is characters of sampled text per second on one A10G GPU.

**The link between the two picks.** All with 64 satellites.

| Second pick | Bits per character | Quality vs BPE |
| --- | --- | --- |
| Host, given the run | 0.5532 | 0.4% worse |
| Run, given the host | 0.5552 | 0.7% worse |
| Independent of the first pick | 0.6910 | 25.4% worse |

**Two controls.** The first shortens training, the second shrinks the model.

| Control | BPE | Chords, 64 satellites | Chords vs BPE |
| --- | --- | --- | --- |
| Same model, 2,247 steps | 0.6249 | 0.6221 | 0.4% better |
| 2 layers, width 128, 10,000 steps | 0.6966 | 0.6997 | 0.5% worse |

At 2,247 steps the chord model has read exactly as much text as BPE reads in 5,000. In the small-model
control, chords with 32 and 128 satellites scored 0.6986 and 0.6989.

**Quality and speed on English web text.** Same model, 5,000 steps each. Trained on four more C4 files
(397 million characters) and scored on the held-out C4 file (97.7 million characters). BPE vocabulary 32,768;
every chord model uses 4,096 runs.

| Model | Parameters | Passes vs BPE | Bits per character | Quality vs BPE | Speed vs BPE |
| --- | --- | --- | --- | --- | --- |
| BPE | 35.9M | 1.00 | 1.4525 | | 1.00x |
| Chords, 64 satellites | 40.6M | 0.62 | 1.4907 | 2.6% worse | 1.53x |
| Chords, 128 satellites | 40.6M | 0.58 | 1.5016 | 3.4% worse | 1.61x |
| Chords, 256 satellites | 40.6M | 0.55 | 1.5146 | 4.3% worse | 1.64x |

Speed here is measured passes per second times the characters per pass of real text. Sampled text would
overstate the chords' lead: the BPE model's samples averaged 3.35 characters per pass, against 4.46 in real
text. Unlike on TinyStories, the chord models trail BPE at every checkpoint during training, by a roughly
steady margin.

What this shows:

- **Same quality for about half the passes, on TinyStories.** At equal training steps, every chord model with
  linked picks is within 1.4% of BPE, in the main run and in both controls, while needing 0.39 to 0.54 of the
  passes.
- **On ordinary English it is a trade, not a free win.** At equal model size, chords generate 1.5x to 1.6x
  faster than BPE but are 2.6% to 4.3% worse. Against a 131,072-token BPE they still need 35% to 43% fewer
  passes.
- **The two halves of a chord depend on each other, and one lookup is enough to capture it.** Picking them
  independently costs 25%. Linking them removes the loss, in either order. The three models also agree with
  each other: the cost of the run picked blind (0.2213) plus the host picked blind (0.4684) is 0.6897, and
  the independent model measures 0.6910.
- **On TinyStories the remaining cost shows up late and stays small.** Chords lead early in training, when
  reading more text per step helps most, and BPE catches up. In the small model, trained until it had nearly
  stopped improving, BPE finished 0.3% to 0.5% ahead.
- **Each pass carries more.** A BPE pass delivers 2.26 bits on TinyStories and 6.48 on web text; a chord pass
  with 64 satellites delivers 5.06 and 10.74.

What it does not show:

- That chords learn more from the same text. Trained on equal text, the chord model is 12.9% worse than
  fully trained BPE. A BPE model given the same reduced number of steps is 13.4% worse, so the gap comes
  from the shorter training. At equal steps a chord model reads about twice as much text; whether it could
  reuse text instead is untested.
- Equal training cost. At these sizes a chord step takes longer to train than a BPE step, because of the
  second output head: 9% to 21% longer at 6 layers and 25% to 51% longer at 2 layers on TinyStories, and 19%
  to 20% longer on web text (wall-clock, one run each). So the comparisons above flatter chords somewhat. By
  FLOP count, a 4,096-run head would add about 2.5% to a model the size of GPT-2 small and under 0.3% to one
  with 7 billion parameters, but the measured overheads here have been 1.4 to 3.2 times what FLOP counts
  predict.
- Speed at equal quality on web text. Closing the 2.6% gap would take a larger chord model, which costs time
  per pass. By FLOP count I estimate that leaves about 1.2x to 1.3x faster generation, but it has not been
  measured.
- That the gaps are real. There is no measure of run-to-run noise, and every difference under about 1% is
  within what one seed can produce.
- Fully converged behaviour. The small models were still improving by about 1% per 2,000 steps at the end.
- Other languages and code. Only English has been counted or trained on.
- That the extra parameters do not matter. Chord models carry 7% to 37% more parameters in the main run, and
  up to 62% more in the small one. They are all lookup tables and the transformer is identical, but it is a
  difference.
- The full speed-up. A chord pass took 1.10x as long as a BPE pass at 6 layers and 1.25x at 2 layers, and
  sampled text has fewer characters per pass than real text. Timing is one sequence at a time with no
  key-value cache.
- Novelty. Multi-word tokens and models with several output heads both exist. This combination has not been
  checked against the literature.
- Anything against other ways of generating faster, such as speculative decoding.

## Abjad

Abjad scripts such as Hebrew and Arabic write mostly consonants and trust the reader to supply the vowels
from context. This experiment applies that to a character-level transformer:

1. A **skeleton model** generates text one character at a time, but only the non-vowel characters.
   `Once upon a time` becomes `nc pn  tm`.
2. A **vocalizer** reads the whole skeleton at once and fills every vowel back in, in a single pass.

The skeleton is about 70% of the original length, so the slow one-character-at-a-time model takes about 30%
fewer steps. The question is how much quality that costs.

### Fairness

Splitting text into a skeleton plus the removed vowels loses nothing, so

    P(text) = P(skeleton) x P(vowels | skeleton)

The two models together are therefore a proper language model, and their combined **bits per character** can
be compared directly with a normal character-level model of the same total size (the baseline).

### Results

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
- Anything about subword tokenizers. All three systems are character-level. Chord tokens grew out of this
  limit: the same idea, applied to whole tokens instead of letters, with no second model.

Example of the vocalizer reading a skeleton from held-out text:

    skeleton:  Sh s nt hppy. Sh sys, "Wht r y dng? Ths rcrd ws  gft frm yr grndm.
    rebuilt:   She is not happy. She says, "What are you doing? Thes record was a gift from your grandma.
    original:  She is not happy. She says, "What are you doing? This record was a gift from your grandma.

## Layout

| File | Purpose |
| --- | --- |
| `bpe.py` | Byte-level BPE in the style of GPT-2, on the `tokenizers` library |
| `chords.py` | Lossless grouping of a token stream into chords, and the inverse |
| `chord_count.py` | Passes per character for BPE and for chords, before training anything |
| `chord_prepare.py` | Tokenizes the text once and saves the BPE and chord streams |
| `chord_model.py` | The transformer whose every position is a chord |
| `chord_train.py` | Trains the BPE baseline or a chord model |
| `chord_evaluate.py` | Passes, bits per character, generation speed and samples for every trained model |
| `modal_chords.py` | Runs the chord experiment and its controls on Modal |
| `abjad.py` | Lossless split of text into skeleton and vowel runs, and the inverse |
| `data.py` | Loads a text file and encodes it for each of the three abjad models |
| `model.py` | One small transformer, causal or bidirectional, and the block both experiments share |
| `train.py` | Trains the abjad models, and holds the training loop both experiments share |
| `evaluate.py` | Bits per character, vocalizer accuracy, generation speed, samples |
| `modal_app.py` | Runs the abjad experiment on Modal |
| `console.py` | Console setup shared by the command-line entry points |

## Running it

Install the dependencies:

    pip install -r requirements.txt

Chord tokens on Modal:

    modal run modal_chords.py::count
    modal run modal_chords.py::count --text c4
    modal run modal_chords.py::prepare
    modal run --detach modal_chords.py::train
    modal run modal_chords.py::evaluate

The controls, after the steps above:

    modal run --detach modal_chords.py::equal_text
    modal run modal_chords.py::evaluate --run equal_text
    modal run --detach modal_chords.py::capacity
    modal run modal_chords.py::evaluate --run small

The web-text quality test, on C4:

    modal run --detach modal_chords.py::prepare_c4
    modal run --detach modal_chords.py::train_c4
    modal run modal_chords.py::evaluate --run c4

Chord tokens locally, with any two UTF-8 text files:

    python chord_prepare.py --train_text data/train.txt --val_text data/input.txt --settings 64:4096
    python chord_train.py
    python chord_train.py --setting k64_r4096
    python chord_evaluate.py

Abjad on Modal (trains the three models in parallel, then evaluates), then its control:

    modal run modal_app.py::main
    modal run modal_app.py::control

Abjad locally, with any UTF-8 text file at `data/input.txt`:

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
