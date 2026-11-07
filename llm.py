"""U2 - Expand terse VQA answers into full sentences with an LLM.
Backends (auto): Gemini (if GEMINI_API_KEY secret set) -> local Qwen2.5-0.5B-Instruct -> rule-based template.
Two prompt techniques are compared: zero_shot and few_shot.
Guardrail: if the LLM sentence does not contain the terse answer, fall back to the template (anti-hallucination)."""
import time
import requests
from config import LLM_MODEL_ID, GEMINI_API_KEY, GEMINI_MODEL, MOCK
from templates import template_expand
from utils import contains_answer

SYSTEM = ("You rewrite a short answer about an image as ONE complete, natural English sentence. "
          "Use ONLY the information in the question and the short answer. Do not add any new details, "
          "objects, colours or guesses. Always state the short answer explicitly.")

FEW_SHOT = [
    ("What color is the circle?", "red", "The circle is red."),
    ("How many squares are there?", "3", "There are 3 squares in the image."),
    ("Is there a triangle in the image?", "no", "No, there is no triangle in the image."),
]


def _user_prompt(q, a):
    return f"Question: {q}\nShort answer: {a}\nOne-sentence answer:"


def build_messages(q, a, technique):
    msgs = [{"role": "system", "content": SYSTEM}]
    if technique == "few_shot":
        for eq, ea, es in FEW_SHOT:
            msgs.append({"role": "user", "content": _user_prompt(eq, ea)})
            msgs.append({"role": "assistant", "content": es})
    msgs.append({"role": "user", "content": _user_prompt(q, a)})
    return msgs


def _clean(text, answer=None):
    """Pick the first usable line. Small instruct models often emit a preamble
    ("Sure, here is the sentence:") on line 1 and the real sentence on line 2, so
    prefer the first non-empty line that actually states the terse answer."""
    lines = [l.strip().strip('"').strip() for l in (text or "").splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return ""
    best = lines[0]
    if answer is not None:
        for l in lines:
            if contains_answer(l, answer):
                best = l
                break
    return " ".join(best.split()[:40])


class Expander:
    def __init__(self):
        self.mock = MOCK
        self._tok = None
        self._lm = None
        self.local_failed = False

    def backend(self):
        if self.mock:
            return "template"
        if GEMINI_API_KEY:
            return "gemini"
        return "local" if not self.local_failed else "template"

    @property
    def name(self):
        b = self.backend()
        return {"gemini": f"Gemini API ({GEMINI_MODEL})", "local": f"{LLM_MODEL_ID} (local)",
                "template": "rule-based templates"}[b]

    def _load_local(self):
        if self._lm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
            self._lm = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID).eval()

    def _gen_local(self, q, a, technique):
        import torch
        self._load_local()
        text = self._tok.apply_chat_template(build_messages(q, a, technique), tokenize=False,
                                             add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt")
        with torch.no_grad():
            out = self._lm.generate(**inputs, max_new_tokens=48, do_sample=False)
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def _gen_gemini(self, q, a, technique):
        msgs = build_messages(q, a, technique)
        prompt = msgs[0]["content"] + "\n\n"
        for m in msgs[1:]:
            prompt += ("" if m["role"] == "user" else "Answer: ") + m["content"] + "\n"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        headers = {"x-goog-api-key": GEMINI_API_KEY, "Content-Type": "application/json"}
        gen = {"temperature": 0.2, "maxOutputTokens": 256,
               # 2.5-series models spend the whole token budget on hidden "thinking"
               # and return a candidate with no text unless thinking is switched off.
               "thinkingConfig": {"thinkingBudget": 0}}
        reason = "unknown"
        for attempt in range(3):
            r = requests.post(url, headers=headers,
                              json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": gen},
                              timeout=30)
            if r.status_code == 400 and "thinkingConfig" in gen:
                gen.pop("thinkingConfig")      # older model: retry without it
                continue
            if r.status_code == 429:           # free-tier rate limit: back off and retry
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            cand = (r.json().get("candidates") or [{}])[0]
            parts = (cand.get("content") or {}).get("parts") or []
            txt = "".join(p.get("text", "") for p in parts).strip()
            if txt:
                return txt
            reason = cand.get("finishReason", "empty response")
            break
        raise RuntimeError(f"Gemini returned no usable text ({reason})")

    def expand(self, question, answer, technique="few_shot"):
        """Returns dict: raw_text, text (after guardrail), backend, raw_contains, fallback_used, latency."""
        t0 = time.time()
        backend = self.backend()
        raw = None
        if backend in ("gemini", "local"):
            try:
                raw = _clean(self._gen_gemini(question, answer, technique) if backend == "gemini"
                             else self._gen_local(question, answer, technique), answer) or None
            except Exception:
                if backend == "local":
                    self.local_failed = True
                raw = None
                backend = "template"
        if raw is None:
            raw = template_expand(question, answer)
            backend = "template" if backend == "template" else backend
        ok = contains_answer(raw, answer)
        final = raw if ok else template_expand(question, answer)
        return {"raw_text": raw, "text": final, "backend": backend, "raw_contains": ok,
                "fallback_used": (not ok), "latency": time.time() - t0}
