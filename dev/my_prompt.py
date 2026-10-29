"""YOUR PART: how we ask Claude for a probability, and how we read the answer.

Two functions to write. llm_prior_test.py calls them for every sampled pair.

subject: dict with normalized_name, provider, release_date, access_date,
         harness, reasoning_effort, harness_version, subject_features_extra
item:    dict with item_content, item_features, interactors, benchmark_id
         (item_content can be very long: use shorten() below)

Ideas to think about while writing the prompt:
  - what should Claude know about the subject? (name, provider, release date, settings)
  - what about the item? (task text, features) How much of it, to keep tokens low?
  - how should Claude answer? (a single number is easiest to read back)
  - Brier punishes confident mistakes: tell Claude to stay away from 0 and 1
  - benchmark_id is anonymous, so it says nothing about the task by itself
"""


import re

MAX_ITEM_CHARS = 4000   # item text is cut to this many characters


def shorten(text, max_chars):
    """Cut long text. Item texts can be 100,000+ characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[cut]"


def build_prompt(subject, item):
    """Return the text we send to Claude for this (subject, item)."""
    # cut the long item text and drop the anonymous benchmark id
    item = dict(item)
    item["item_content"] = shorten(item["item_content"], MAX_ITEM_CHARS)
    item.pop("benchmark_id", None)

    prompt = """
    You are a LLM expert. You know the capabilities of many LLMs, and you can predict how well they will do on a task. 
    You are given a subject (a LLM) and an item (a task). Your job is to predict the probability that the subject will answer the item correctly.
    Subject information:""" + str(subject) + """
    Item information:""" + str(item) + """

    Please provide a single number between 0 and 1, representing the probability that the subject will answer the item correctly. 
    Avoid giving a probability of exactly 0 or 1, as we want to avoid overconfidence. Make sure to consider the subject's capabilities, the task's difficulty, and any other relevant information when making your prediction.
    Your answer should be a single number, with no additional text or explanation.
    """
    return prompt.strip()


def parse_probability(reply):
    """Turn Claude's reply into a float between 0 and 1. Return None if you cannot."""
    if reply is None:
        return None
    # first number in the text, like 0.7, .7 or 65
    match = re.search(r"\d*\.?\d+", reply)
    if match is None:
        return None
    number = float(match.group())
    # 65 or 65% means 0.65
    if number > 1:
        number = number / 100
    if number < 0 or number > 1:
        return None
    return number
