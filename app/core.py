"""Your code, one function per milestone. Each builds on the WordFlow engine in wordflow/."""

from pathlib import Path

from wordflow.corpus import corpus_from_file
from wordflow.generation import TextGenerator
from wordflow.ngram_model import NGramModel
from wordflow.predictor import NextWordPredictor
from wordflow.preprocessing import tokenize


# --- Milestone 1: Prepare a text corpus of short stories.
# Hint: Use text preprocessing and tokenization to prepare your corpus.
ROOT = Path(__file__).resolve().parent.parent
FANTASY_CORPUS = ROOT / "corpus" / "fantasy" / "fantasy.txt"


def prepare_corpus(path=FANTASY_CORPUS) -> list[list[str]]:
    """Load a text file, clean it, split it into sentences and tokenize each one.

    Returns one list of lower-case word tokens per sentence. The default corpus is
    built from eight public-domain fantasy books (see scripts/build_fantasy_corpus.py).
    """
    corpus = corpus_from_file(path)
    return [tokenize(sentence) for sentence in corpus.sentences]


# --- Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
# Hint: Apply n-gram counting and maximum-likelihood estimation to calculate probabilities.
def train_model(sentences, order: int = 3, min_count: int = 1) -> NGramModel:
    """Count unigrams, bigrams and trigrams (every n up to ``order``) over the sentences."""
    return NGramModel(order, min_count).fit(sentences)


def mle_probability(model: NGramModel, word: str, context=()) -> float:
    """Maximum-likelihood P(word | context) = C(context, word) / C(context).

    Words never seen after ``context`` get probability exactly 0.
    """
    return model.probability(word, context, k=0.0)


# --- Milestone 3: Add add-k smoothing to handle unseen n-grams.
# Hint: Implement add-k smoothing to manage out-of-vocabulary n-grams.
def smoothed_probability(model: NGramModel, word: str, context=(), k: float = 1.0) -> float:
    """Add-k estimate P(word | context) = (C(context, word) + k) / (C(context) + k * V).

    Every word in the vocabulary gets a non-zero probability, so unseen n-grams no
    longer break the model. ``k = 1`` is Laplace smoothing; small k trusts the counts more.
    """
    return model.probability(word, context, k=k)


# --- Milestone 4: Develop a function to generate text by sampling from the model.
# Hint: Create a sampling function to generate text based on your model's probabilities.
def generate_text(model: NGramModel, seed_text: str = "", max_words: int = 30,
                  temperature: float = 1.0, seed: int | None = None, k: float = 0.0) -> str:
    """Continue ``seed_text`` by sampling one word at a time from the model.

    The same ``seed`` always gives the same text. ``temperature`` below 1 makes the
    choices more conservative, above 1 more adventurous.
    """
    result = TextGenerator(model).generate_text(
        seed_text, max_words=max_words, temperature=temperature, seed=seed, k=k,
        stop_at_eos=False,
    )
    return result.text


# --- Milestone 5: Create a CLI interface for users to input text and get suggestions.
# Hint: Build a command-line interface to take user input and display suggestions.
def suggest_next_words(model: NGramModel, text: str, top_k: int = 5,
                       k: float = 0.01) -> list[tuple[str, float]]:
    """The ``top_k`` most probable next words for ``text`` as (word, probability) pairs.

    The command line (``python -m app.cli write --genre fantasy``) wraps this in an
    interactive story-writing session, see ``wordflow/writer.py``.
    """
    result = NextWordPredictor(model).suggest_next(text, top_k=top_k, k=k)
    return [(s.word, s.probability) for s in result.suggestions]
