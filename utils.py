"""Small text helpers shared by the app and the evaluation."""
import re

NUM_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
ARTICLES = {"a", "an", "the"}


def normalize(s: str) -> str:
    """Lowercase, strip punctuation/articles, map number words to digits."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    toks = [NUM_WORDS.get(t, t) for t in s.split() if t not in ARTICLES]
    return " ".join(toks)


def is_correct(pred: str, gt: str) -> bool:
    """Prediction is correct if, after normalisation, it equals the ground truth
    or contains the ground truth as a whole word."""
    p, g = normalize(pred), normalize(gt)
    return bool(g) and (p == g or g in p.split())


def contains_answer(expansion: str, answer: str) -> bool:
    """Faithfulness check: does the expanded sentence still state the terse answer?"""
    toks = set(normalize(expansion).split())
    a = normalize(answer)
    if not a:
        return False
    if a == "yes":
        return "yes" in toks
    if a == "no":
        return bool(toks & {"no", "not", "isnt", "arent", "dont", "doesnt"})
    return all(t in toks for t in a.split())
