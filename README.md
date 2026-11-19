# Creative Writing Prompter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/mqjv39kx74/badge.svg)](https://ask.thili.ai/b/mqjv39kx74)

The Creative Writing Prompter is a CLI tool designed for aspiring writers. It helps overcome writer's block by offering context-aware next-word suggestions, sampling from a genre corpus using a custom-built next-word prediction engine.

## What you are building

**Goal.** You will build a command-line interface tool that provides context-aware next-word suggestions for aspiring writers, leveraging a next-word prediction engine to generate text by sampling based on computed n-gram probabilities.

**Done when.** The project is done when your CLI tool can take user input, process it with an n-gram model, and suggest next words by sampling from probabilities derived from a short story corpus. It should successfully handle unseen n-grams using add-k smoothing.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Prepare a text corpus of short stories.**  
      Hint: Use text preprocessing and tokenization to prepare your corpus.
- [ ] **2. Implement n-gram counting and maximum-likelihood probabilities.**  
      Hint: Apply n-gram counting and maximum-likelihood estimation to calculate probabilities.
- [ ] **3. Add add-k smoothing to handle unseen n-grams.**  
      Hint: Implement add-k smoothing to manage out-of-vocabulary n-grams.
- [ ] **4. Develop a function to generate text by sampling from the model.**  
      Hint: Create a sampling function to generate text based on your model's probabilities.
- [ ] **5. Create a CLI interface for users to input text and get suggestions.**  
      Hint: Build a command-line interface to take user input and display suggestions.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- (none)
