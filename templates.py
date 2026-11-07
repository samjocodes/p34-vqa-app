"""U1 - Question templates + question-type detection + deterministic sentence templates."""
import re

# (category, template text, slot name or None, example slot value)
TEMPLATES = [
    ("Colour",      "What color is the {object}?",                  "object",  "circle"),
    ("Counting",    "How many {objects} are there?",                "objects", "squares"),
    ("Existence",   "Is there a {object} in the image?",            "object",  "triangle"),
    ("Identity",    "What shape is the {color} object?",            "color",   "blue"),
    ("Location",    "Where is the {object}?",                       "object",  "circle"),
    ("Scene",       "What is the main object in this image?",       None,      ""),
    ("Lighting",    "Is it dark in this image?",                    None,      ""),
    ("Scene",       "What is in the center of the image?",          None,      ""),
]


def template_labels():
    return [f"[{c}] {t}" for c, t, _, _ in TEMPLATES]


def get_template(label: str):
    for (c, t, slot, ex) in TEMPLATES:
        if f"[{c}] {t}" == label:
            return {"category": c, "text": t, "slot": slot, "example": ex}
    return None


def build_question(label: str, slot_value: str = "") -> str:
    """Fill a template. Raises ValueError if a required slot is empty."""
    tpl = get_template(label)
    if tpl is None:
        raise ValueError("Unknown template")
    if tpl["slot"] is None:
        return tpl["text"]
    val = (slot_value or "").strip()
    if not val:
        raise ValueError(f"Please fill the blank: {tpl['slot']} (e.g. '{tpl['example']}')")
    return tpl["text"].replace("{" + tpl["slot"] + "}", val)


def question_type(q: str) -> str:
    ql = (q or "").strip().lower()
    if ql.startswith("how many"):
        return "count"
    if re.match(r"^(is|are|does|do|can|was|were)\b", ql):
        return "yesno"
    if re.search(r"\bwhat colou?r\b", ql):
        return "color"
    if re.search(r"\bwhat shape\b", ql):
        return "shape"
    if ql.startswith("where"):
        return "location"
    return "other"


def template_expand(question: str, answer: str) -> str:
    """Deterministic rule-based sentence (also the safety fallback for the LLM)."""
    q = (question or "").strip().rstrip("?").strip()
    a = (answer or "").strip()
    al = a.lower()
    m = re.search(r"what colou?r (?:is|are) (?:the |a |an )?(.+)$", q, re.I)
    if m:
        return f"The {m.group(1)} is {al}."
    m = re.search(r"how many (.+?) (?:are|is) (?:there|visible|present)(?: in the (?:image|picture|photo))?$", q, re.I) \
        or re.search(r"how many (.+)$", q, re.I)
    if m:
        obj = m.group(1)
        if al in ("1", "one"):
            obj = obj[:-1] if obj.endswith("s") else obj
            return f"There is 1 {obj}."
        return f"There are {al} {obj}."
    m = re.search(r"^(?:is|are) there (?:a |an |any )?(.+?)(?: in the (?:image|picture|photo))?$", q, re.I)
    if m:
        x = m.group(1)
        if al == "yes":
            return f"Yes, there is a {x} in the image."
        if al == "no":
            return f"No, there is no {x} in the image."
    m = re.search(r"what shape is (?:the )?(.+)$", q, re.I)
    if m:
        return f"The {m.group(1)} is a {al}."
    if question_type(q) == "yesno":
        return f"{a.capitalize()}, that is what the image suggests."
    return f"The answer is {a}."
