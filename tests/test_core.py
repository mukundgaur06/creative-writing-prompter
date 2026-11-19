"""One test per milestone in README.md."""
import pytest

from app.core import FANTASY_CORPUS, mle_probability, prepare_corpus, smoothed_probability, train_model
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
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Create a CLI interface for users to input text and get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
