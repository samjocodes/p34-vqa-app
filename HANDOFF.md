# HANDOFF – read this first (for any agent/human continuing this project)

You are continuing a student's university assignment from a previous session. Everything needed is in this folder; you do **not** need the earlier chat.

## 1. Who / what / when
- Student: **Saumya Joshi**, enrollment **ADT23SOCB0997**, **Roll 34**, class **AIA-05 Batch B**.
- Assignment: Generative AI "60 LLM Mini-Projects", roll 34 → **P34 · Visual Question Answering App**. Exact requirements: `docs/REQUIREMENTS.md`. Requirement→evidence table: `docs/REQUIREMENTS_TRACE.md`.
- **Deadline: Saturday 7 November 2026.** Individual work. Submit in drive folder `34_Saumya_Joshi`: PDF report, live app URL + GitHub link, 2–3 min demo video. Also public GitHub repo with README + model card.
- The student has little time and low GenAI background: do the work for them, give click-by-click instructions, keep answers short. Be honest; never invent results.

## 2. Current state (end of session 2 — reviewed and patched)

Session-2 review found no blocking bugs; all 20 benchmark ground-truth labels were re-verified
against the rendered images. Four fixes were applied:
1. `llm.py::_clean` — kept only line 1 of the LLM output, so a preamble line
   ("Sure, here is the sentence:") destroyed the answer and forced the template fallback on
   every pair, making the zero-shot/few-shot comparison degenerate. Now picks the first
   non-empty line that actually states the terse answer.
2. `llm.py::_gen_gemini` — Gemini 2.5 models spend the whole token budget on hidden thinking
   and return a candidate with no `parts`, which raised `KeyError` on every call. Now sends
   `thinkingConfig.thinkingBudget = 0` (retrying without it for older models), parses the
   response defensively, and backs off on HTTP 429.
3. `vqa_model.py::answer` — an exception while extracting probabilities discarded an already
   decoded answer and ran inference a second time. Now the answer is kept.
4. `synth.py::TESTSET_DIR` — now prefers the `testset/` folder committed next to the code
   (so the report figures are the exact committed images), falling back to the temp dir.

Also added: `smoke_test.py` (pre-flight model check), `P34_colab_V2.ipynb`,
`results/` folder, `LINKS.txt`, rewritten `docs/DEPLOY_GUIDE.md` with a no-git browser-upload path.

## 3. Earlier state (end of session 1)
| Item | State |
|---|---|
| Code (app, models, guardrails, generator, evaluation, report builder) | Written. Offline unit tests pass (`VQA_MOCK=1 python tests/test_core.py`, 5/5). Gradio UI tested with Gradio 6.31 using mock models. |
| Benchmark | 20 image–question pairs + 10 bias silhouettes pre-generated in `testset/` (ground truth verified programmatically). |
| Real models (BLIP-VQA, Qwen2.5-0.5B, MiniLM, SD-Turbo) | **NEVER RUN** – the build sandbox had no access to huggingface.co. Code paths are written carefully but unverified. |
| Real evaluation numbers / PDF report | **Not produced.** Must come from a real run (Evaluation tab or `python vqa_eval.py`). No numbers exist anywhere in the repo. |
| Deployment (HF Space, GitHub) | **Not done** – needs the student's accounts. Guide: `docs/DEPLOY_GUIDE.md`. |
| Demo video | Not done – script in `docs/DEMO_VIDEO_SCRIPT.md`. |
| Submission | Not done – `docs/SUBMISSION_CHECKLIST.md`. |

## 3b. Architecture / file map
```
app.py          Gradio UI: tabs Ask / Test-image generator / Evaluation+Report / About
config.py       student info, model IDs, thresholds, env (GEMINI_API_KEY, GEMINI_MODEL, VQA_MOCK)
templates.py    U1 templates, question_type(), template_expand() (rule-based sentence; also LLM fallback)
llm.py          U2 Expander: Gemini (if secret) -> local Qwen2.5-0.5B-Instruct -> template; zero_shot/few_shot; faithfulness fallback
vqa_model.py    U3 BLIP-VQA wrapper: answer, mean token prob (confidence), top-5 first-token probs
bert_check.py   Layer III BERT-family MiniLM cosine similarity
synth.py        U4 procedural scenes (clean/lowlight/cluttered), BENCHMARK list, enhance_low_light(), optional SD-Turbo
guardrails.py   Layer V: sensitive-question refusal, injection check, image checks, low-confidence flag
vqa_eval.py     U5 evaluation generator (+ CLI) -> results dict; summary_markdown()
report.py       builds the PDF from results (all numbers injected, none hard-coded)
lifecycle.py    draws the lifecycle diagram PNG (temp file, used in the report)
testset/        20 benchmark PNGs + testset.json + bias/ (10 silhouettes)   (regenerate: python synth.py)
tests/test_core.py  offline tests
README.md (has HF YAML front-matter), MODEL_CARD.md, requirements.txt, LICENSE
```
Evaluation design: 20 pairs = 8 clean, 6 low-light (14% brightness + noise), 6 cluttered (26 scribbles). Correct = normalised equality or whole-word containment (`utils.is_correct`). Low-light is also re-run after `enhance_low_light` (before/after table). Prompt comparison = zero-shot vs few-shot faithfulness. Bias test = gendered answers on 10 neutral silhouettes. Guardrail test = 10 sensitive vs 10 benign questions.

## 4. Known risks (check these first if something breaks) 
1. **BLIP confidence code** (`vqa_model.py`) uses `generate(..., output_scores=True, return_dict_in_generate=True)`; it falls back to plain generation (confidence None) on exceptions. If confidence is always `n/a`, debug there.
2. **transformers version**: pinned `>=4.44,<5`. If the Space build fails on versions, read Logs and adjust `requirements.txt`.
3. **torch CPU index line** at the top of `requirements.txt`: if the build fails on it, delete that line (torch will still install, just bigger).
4. **Qwen local load** may be slow/fail on free CPU: `Expander` then silently uses templates (`llm_backend` = "rule-based templates" in the report, which adds an explicit note). Acceptable fallback, but try to fix (check memory/Logs) so the LLM comparison is real. Alternative: set `GEMINI_API_KEY` secret (model via `GEMINI_MODEL`, default `gemini-2.5-flash-lite`; Gemini model names change often – verify current free model at ai.google.dev).
5. **Evaluation tab duration** on free CPU (~40 BLIP calls + 40 LLM calls + 10 bias). If the browser times out use Plan B.
6. **SD-Turbo** is optional/heavy; failure is handled with a message. Layer IV is already met by the procedural generator, but SD-Turbo gives a "real" generative-model image feature – verify once, don't block on it.
7. Accuracy may be modest on synthetic shapes (esp. counting, dark images). That is **fine** – the assignment wants measured results and failure analysis. Do **not** alter ground truth or hand-edit numbers to look better. Improving prompts/preprocessing is allowed if you re-run the evaluation honestly.
8. HF free Spaces sleep after inactivity; keep the Space running/awake until marks are out.

## 5. Plan B – run evaluation outside the Space (Google Colab, free)
```python
!git clone https://github.com/<user>/p34-vqa-app.git
%cd p34-vqa-app
!pip install -q -r requirements.txt
import os
os.environ["LIVE_URL"] = "https://huggingface.co/spaces/<user>/p34-vqa-app"
os.environ["GITHUB_URL"] = "https://github.com/<user>/p34-vqa-app"
!python vqa_eval.py
```
Outputs in `outputs/`: `results.json`, `results_summary.md`, `P34_report_Saumya_Joshi.pdf`. Download them, put in a `results/` folder in the GitHub repo.

## 6. What to do next (in order)
1. 👤 Student deploys (DEPLOY_GUIDE Parts 1–2). If the build fails, get Logs → fix → re-upload.
2. Verify the real app: one VQA question works; sensitive question refused; evaluation completes.
3. Student downloads PDF + results.json + results_summary.md → GitHub `results/`; fills live URL in README.
4. Review the PDF: real numbers present, failure case present, backend line shows the Qwen/Gemini model (not "rule-based").
5. 👤 Student records video (script provided) and submits (checklist).
Quick sanity commands: `VQA_MOCK=1 python tests/test_core.py` · `python synth.py` · `python lifecycle.py`.

## 7. Rules for the next agent
- Never fabricate or hard-code results; `VQA_MOCK=1` is test-only and its output must never be shown as real.
- Keep API keys out of the repo (secrets only). Keep one core feature working before adding extras.
- If you change anything that affects requirements, update `docs/REQUIREMENTS_TRACE.md`.
- The student's name/IDs live in `config.py` (STUDENT) – used by app header and report.

## 8. Paste-ready prompt for a fresh agent
> I am Saumya Joshi (Roll 34, AIA-05 Batch B). Read HANDOFF.md, docs/REQUIREMENTS.md and docs/REQUIREMENTS_TRACE.md in the attached project and continue from "What to do next". The deadline is Saturday 7 Nov 2026. Here is the current problem/log/state: <paste Space build logs or error here>. Fix it and tell me exactly what to click.
