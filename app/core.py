"""Your code, one function per milestone. Each builds on the WordFlow engine in wordflow/."""

from pathlib import Path

from wordflow.corpus import corpus_from_file
from wordflow.ngram_model import NGramModel
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
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Develop a function to generate text by sampling from the model.
# Hint: Create a sampling function to generate text based on your model's probabilities.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Create a CLI interface for users to input text and get suggestions.
# Hint: Build a command-line interface to take user input and display suggestions.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
