"""WordFlow -- a statistical next-word prediction engine built from first principles.

Only the Python standard library is used. Typical use::

    from wordflow import NGramModel, NextWordPredictor, tokenize_sentences

    sentences = tokenize_sentences("I love Python\\nI love machine learning\\nI love AI")
    model = NGramModel(order=3).fit(sentences)
    NextWordPredictor(model).suggest_next("I love", top_k=3, smoothing=False)
"""

from .corpus import Corpus, CorpusError, build_corpus, corpus_from_file, split_train_test
from .evaluation import EvaluationResult, compare_models, evaluate, score_sentence
from .ngram_model import NGramModel
from .predictor import NextWordPredictor, PredictionResult, Suggestion
from .preprocessing import (
    BOS,
    EOS,
    UNK,
    Vocabulary,
    pad_sentence,
    split_sentences,
    tokenize,
    tokenize_sentences,
)
from .smoothing import add_k_probability, mle_probability

__version__ = "1.0.0"

__all__ = [
    "BOS",
    "Corpus",
    "CorpusError",
    "EOS",
    "EvaluationResult",
    "NGramModel",
    "NextWordPredictor",
    "PredictionResult",
    "Suggestion",
    "UNK",
    "Vocabulary",
    "add_k_probability",
    "build_corpus",
    "compare_models",
    "corpus_from_file",
    "evaluate",
    "mle_probability",
    "pad_sentence",
    "score_sentence",
    "split_sentences",
    "split_train_test",
    "tokenize",
    "tokenize_sentences",
]
