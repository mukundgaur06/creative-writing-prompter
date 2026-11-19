"""One test per milestone in README.md."""
import pytest

from app.core import FANTASY_CORPUS, generate_text, mle_probability, prepare_corpus, smoothed_probability, suggest_next_words, train_model
from wordflow.preprocessing import tokenize

TOY = [tokenize(s) for s in ("I love Python", "I love machine learning", "I love AI")]


# Milestone 1: Prepare a text corpus of short stories.
def test_milestone_1():
    """The fantasy corpus loads as lower-case tokenized sentences."""
    sentences = prepare_corpus(FANTASY_CORPUS)
    assert len(sentences) > 10_000
    words = [word for sentence in sentences for word in sentence]
    assert all(word == word.lower() for word in words)
    assert "dragon" in words and "gutenberg" not in words


# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
def test_milestone_2():
    """Counting gives exact maximum-likelihood probabilities on a tiny corpus."""
    model = train_model(TOY)
    assert mle_probability(model, "love", ("i",)) == 1.0
    assert mle_probability(model, "python", ("i", "love")) == pytest.approx(1 / 3)
    assert mle_probability(model, "dragon", ("i", "love")) == 0.0


# Milestone 3: Add add-k smoothing to handle unseen n-grams.
def test_milestone_3():
    """Add-k gives unseen words probability mass and still sums to one."""
    model = train_model(TOY)
    assert smoothed_probability(model, "dragon", ("i", "love"), k=1) > 0
    total = sum(model.distribution(("i", "love"), k=1).values())
    assert total == pytest.approx(1.0)
    # small k trusts the counts more, so a seen word keeps more of the mass
    assert smoothed_probability(model, "python", ("i", "love"), k=0.1) > \
        smoothed_probability(model, "python", ("i", "love"), k=1)


# Milestone 4: Develop a function to generate text by sampling from the model.
def test_milestone_4():
    """Sampling is reproducible for a seed and continues the seed text."""
    model = train_model(TOY)
    a = generate_text(model, "i love", max_words=5, seed=1)
    b = generate_text(model, "i love", max_words=5, seed=1)
    assert a == b
    assert a.lower().startswith("i love")
    assert len(a.split()) > 2


# Milestone 5: Create a CLI interface for users to input text and get suggestions.
def test_milestone_5(tmp_path, capsys):
    """Suggestions are ranked, and the CLI predicts from a saved model."""
    model = train_model(TOY)
    suggestions = suggest_next_words(model, "i love", top_k=3)
    assert len(suggestions) == 3
    probabilities = [p for _, p in suggestions]
    assert probabilities == sorted(probabilities, reverse=True)

    from app.cli import main

    path = tmp_path / "model.json"
    model.save(path)
    assert main(["predict", "i love", "--model", str(path)]) == 0
    assert "Input:" in capsys.readouterr().out
