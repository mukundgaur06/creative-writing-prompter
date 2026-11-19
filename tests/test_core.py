"""One test per milestone in README.md."""
import pytest

from app.core import FANTASY_CORPUS, prepare_corpus


# Milestone 1: Prepare a text corpus of short stories.
def test_milestone_1():
    """The fantasy corpus loads as lower-case tokenized sentences."""
    sentences = prepare_corpus(FANTASY_CORPUS)
    assert len(sentences) > 10_000
    words = [word for sentence in sentences for word in sentence]
    assert all(word == word.lower() for word in words)
    assert "dragon" in words and "gutenberg" not in words


# Milestone 2: Implement n-gram counting and maximum-likelihood probabilities.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Add add-k smoothing to handle unseen n-grams.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Develop a function to generate text by sampling from the model.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Create a CLI interface for users to input text and get suggestions.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")
