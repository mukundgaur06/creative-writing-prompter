# Creative Writing Prompter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/mqjv39kx74/badge.svg)](https://ask.thili.ai/b/mqjv39kx74)
[![tests](https://github.com/mukundgaur06/creative-writing-prompter/actions/workflows/ci.yml/badge.svg)](https://github.com/mukundgaur06/creative-writing-prompter/actions/workflows/ci.yml)

A command-line tool for aspiring writers. It suggests the next word of your story,
learned from a corpus of fantasy tales, to help you past writer's block. The language model
is built from scratch (counting n-grams, maximum-likelihood estimation, add-k smoothing,
sampling), with no neural networks, no pretrained models, no API keys and **no
third-party dependencies**.

```
$ python -m app.cli write "once upon a" --genre fantasy

  once upon a

  1. time              16.61%
  2. stone              1.92%
  3. beam               1.28%
  4. bush               1.28%
  5. couch              1.28%
  [most likely suggestions - ? for help]
> 1
```

## What you are building

**Goal.** A command-line interface tool that provides context-aware next-word suggestions
for aspiring writers, leveraging a next-word prediction engine to generate text by sampling
based on computed n-gram probabilities.

**Done when.** The CLI can take user input, process it with an n-gram model, and suggest
next words by sampling from probabilities derived from a short story corpus. It handles
unseen n-grams using add-k smoothing.

## Milestones

- [x] **1. Prepare a text corpus of short stories.**
      `scripts/build_fantasy_corpus.py` builds `corpus/fantasy/fantasy.txt` from eight
      public-domain Project Gutenberg books (about 430,000 words); preprocessing and
      tokenization are in `wordflow/preprocessing.py` and `wordflow/corpus.py`.
- [x] **2. Implement n-gram counting and maximum-likelihood probabilities.**
      `wordflow/ngram_model.py` and `mle_probability` in `wordflow/smoothing.py`.
- [x] **3. Add add-k smoothing to handle unseen n-grams.**
      `add_k_probability` in `wordflow/smoothing.py`, with backoff to shorter contexts in
      `wordflow/predictor.py`.
- [x] **4. Develop a function to generate text by sampling from the model.**
      `TextGenerator.generate_text` in `wordflow/generation.py`: seeded, with temperature.
- [x] **5. Create a CLI interface for users to input text and get suggestions.**
      `python -m app.cli write` (`wordflow/writer.py`), plus `predict`, `generate`,
      `train`, `evaluate`.

`app/core.py` has one small function per milestone and `tests/test_core.py` one test each.

## Run it

You need Python 3.10 or newer (CI uses 3.12) and git. There is nothing to install to run it;
`pytest` is only needed for the tests.

```bash
git clone https://github.com/mukundgaur06/creative-writing-prompter.git
cd creative-writing-prompter

python -m app.cli hello                        # check that everything is wired up
python -m app.cli write --genre fantasy        # interactive story writing
python -m app.cli write "the old wizard" --genre fantasy --creative

pip install -r requirements.txt                # pytest, for the tests only
pytest
```

The first `write --genre fantasy` trains a trigram model on the fantasy corpus (a few
seconds) and caches it in `models/`, which is not committed. The corpus itself is committed;
to rebuild it from Project Gutenberg run `python scripts/build_fantasy_corpus.py`.

### Writing session

| Type | What it does |
|---|---|
| `1`-`9` | accept that suggestion |
| any text | add your own words to the story |
| `r` | new suggestions (a fresh draw in creative mode) |
| `m` | switch between **most likely** and **creative** suggestions |
| `a 20` | let the model write the next 20 words |
| `u` | undo the last step |
| `s story.txt` | save the story |
| `?` / `q` | help / quit and print the story |

- **Most likely** mode lists the top-N words by add-k probability. **Creative** mode draws
  N words at random from the same distribution (flattened by `--temperature`), so repeated
  suggestions differ.
- `write` uses add-k with `k = 0.01`. `k = 1` is far too strong for a 7,000-word vocabulary
  (see the results below). `a` samples only from continuations seen in the corpus, because
  add-k would hand probability to the whole vocabulary after a sparse context.

### Other commands

```bash
python -m app.cli predict "the dragon" --model models/fantasy_model.json --k 0.01
python -m app.cli generate "the princess" --model models/fantasy_model.json --seed 7 --keep-going
python -m app.cli evaluate --test corpus/fantasy/test.txt --model models/fantasy_model.json --k 0.01
python -m app.cli demo                    # split -> train -> predict -> generate -> evaluate (sample corpus)
```

`train`, `split` and `evaluate` read `corpus/` by default; every path can be changed with
`--corpus`, `--train`, `--test`, `--model`, `--out-dir`. Run `python -m app.cli --help` for
everything.

## How it works

1. **Preprocess.** Lower-case, normalise apostrophes and quotes, split into sentences, pad
   each with `<s>` / `</s>`, map rare words to `<unk>`.
2. **Count.** For every order up to 3, count each n-gram. No n-gram spans two sentences.
3. **Estimate.** MLE: `P(w | ctx) = C(ctx, w) / C(ctx)`. Add-k:
   `P(w | ctx) = (C(ctx, w) + k) / (C(ctx) + k V)`, so unseen n-grams get a small non-zero
   probability. If the whole context was never seen, back off to a shorter one.
4. **Suggest and sample.** Rank candidates by probability, or draw words with probability
   proportional to `P ** (1 / temperature)`.

The maths, with worked examples, is in [MATHEMATICS.md](MATHEMATICS.md); the design is in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Results on the fantasy corpus

Held-out evaluation: 21,751 sentences split 80/20 (seed 42), so 17,430 training and 4,321
test sentences, 91,789 test tokens, vocabulary 7,357 (words seen at least twice, 2.15% of
test tokens are out of vocabulary). Perplexity, lower is better
(`python -m app.cli evaluate ... --min-count 2`):

| Model | k = 1 | k = 0.1 | k = 0.01 |
|---|---:|---:|---:|
| Bigram, add-k | 746 | 314 | **229** |
| Trigram, add-k | 3,212 | 1,592 | 951 |
| Trigram, add-k + backoff | 2,352 | 980 | 535 |

For reference, a unigram model scores 450, and plain MLE bigram and trigram models score
infinite perplexity.

- **Why MLE is infinite.** 17,178 of the 91,789 test tokens (bigram) and 52,088 (trigram)
  were never seen after their context, so they get probability 0. Add-k removes every one of
  those zeros.
- **Why small k wins.** With about 7,000 words, `k = 1` moves most of the probability onto
  words that were never seen. Shrinking `k` trusts the counts more.
- **Honest caveat.** The bigram model beats the trigram on this corpus: about 345,000
  training words are too few for trigrams to be reliable (data sparsity), even with backoff.
  `k` was chosen by looking at this same test split; a separate validation split would be
  cleaner.

## Project layout

```
app/                 cli.py (python -m app.cli) and core.py (one function per milestone)
wordflow/            the engine, standard library only
  preprocessing.py     normalise, split sentences, tokenize, pad, Vocabulary
  corpus.py            loading, cleaning, deterministic train/test split
  ngram_model.py       NGramModel: counting, lookups, backoff, save/load
  smoothing.py         MLE and add-k, the only place counts become probabilities
  predictor.py         suggest_next()
  generation.py        generate_text(), temperature, sampling
  evaluation.py        cross-entropy, perplexity, OOV, model comparison
  writer.py            WritingSession and the interactive loop
  service.py, server.py   the optional dashboard backend
main.py              command-line sub-commands (called by app/cli.py)
dashboard.py         optional browser dashboard: python dashboard.py
web/                 dashboard front end (no CDN, works offline)
corpus/              corpus.txt (small sample) and fantasy/ (the story corpus)
scripts/             build_fantasy_corpus.py
tests/               pytest suite (also plain unittest)
```

## Tests

```bash
pytest            # 112 tests: milestones, n-gram maths, smoothing, sampling, writer, dashboard API
```

GitHub Actions runs the same on every push (`.github/workflows/ci.yml`).

## Optional: the dashboard

```bash
python dashboard.py --open      # http://127.0.0.1:8501
```

An 8-page browser lab on the small sample corpus: live predictions, an n-gram explorer, a
probability laboratory with worked MLE and add-k calculations, a text-generation lab with a
step-by-step trace, held-out evaluation, and a corpus upload page. It is an extra; the
command line is the product.

## Corpus and licensing

The fantasy corpus is text from public-domain books downloaded from Project Gutenberg
(the Gutenberg licence header and footer are stripped): *The Wonderful Wizard of Oz* and
*The Marvelous Land of Oz* (L. Frank Baum), *Alice's Adventures in Wonderland* and
*Through the Looking-Glass* (Lewis Carroll), *Peter Pan* (J. M. Barrie), *Grimms' Fairy
Tales*, *The Blue Fairy Book* (Andrew Lang) and *The Princess and the Goblin* (George
MacDonald). `corpus/corpus.txt` is a small original sample written for this project.

## Limitations

- A trigram sees only two previous words, so stories drift: sentences are locally plausible
  but have no plot, and the model has no notion of grammar or meaning.
- The vocabulary is closed: every unseen word becomes the same `<unk>`.
- Add-k is a crude smoother. Kneser-Ney or interpolation with tuned weights would do better.
- The corpus is old-fashioned fairy-tale English, so the suggestions are too.

## Stretch ideas

- More genres: put a `corpus/<genre>/<genre>.txt` in place and run `write --genre <genre>`.
- Kneser-Ney smoothing, or interpolation tuned on a validation split.
- Suggest whole phrases with beam search instead of single words.
