"""Layer V - safety guardrails (input checks, sensitive-attribute refusal, confidence flag)."""
import re
from PIL import Image
from config import MAX_QUESTION_CHARS, MAX_IMAGE_SIDE, LOW_CONF_THRESHOLD

# Questions asking the model to infer protected / sensitive attributes or identify people.
SENSITIVE = re.compile(
    r"\b(gender|sex|sexual|sexuality|orientation|gay|lesbian|transgender|race|racial|ethnic\w*|"
    r"religio\w*|caste|nationality|disabilit\w*|disabled|diseas\w*|illness|pregnan\w*|"
    r"how old|age of|attractive|ugly|beautiful|rich or poor|criminal|terrorist|"
    r"man or (?:a )?woman|woman or (?:a )?man|male or female|boy or (?:a )?girl|"
    r"who is (?:this|that|he|she|the person)|identify (?:this|the|that) person|"
    r"name of (?:this|the|that) person|what is (?:his|her|their) name)\b", re.I)

INJECTION = re.compile(r"(ignore (?:all |the )?(?:previous|above) (?:instructions|prompt)|system prompt|"
                       r"jailbreak|reveal your instructions)", re.I)

REFUSAL = ("I can't answer that. This app does not infer sensitive personal attributes "
           "(such as gender, age, race, religion, health) or identify people from images, "
           "because such guesses are unreliable and can be harmful. "
           "Try a question about objects, colours, counts or positions instead.")


def check_question(q: str):
    """Return (allowed: bool, message: str)."""
    q = (q or "").strip()
    if not q:
        return False, "Please type or build a question first."
    if len(q) > MAX_QUESTION_CHARS:
        return False, f"Question is too long (max {MAX_QUESTION_CHARS} characters)."
    if INJECTION.search(q):
        return False, "That looks like an instruction to the system rather than a question about the image."
    if SENSITIVE.search(q):
        return False, REFUSAL
    return True, ""


def prepare_image(img):
    """Validate and downscale an uploaded image. Returns (img_rgb, message)."""
    if img is None:
        return None, "Please upload an image first."
    if not isinstance(img, Image.Image):
        return None, "Unsupported image input."
    img = img.convert("RGB")
    w, h = img.size
    if w < 16 or h < 16:
        return None, "Image is too small."
    if max(w, h) > MAX_IMAGE_SIDE:
        s = MAX_IMAGE_SIDE / max(w, h)
        img = img.resize((int(w * s), int(h * s)))
    return img, ""


def confidence_flag(conf):
    if conf is None:
        return ""
    if conf < LOW_CONF_THRESHOLD:
        return ("Low confidence: the model is not sure about this answer. "
                "Treat it as a guess, especially for dark or cluttered images.")
    return ""
