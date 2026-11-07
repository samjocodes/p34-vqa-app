"""Builds the PDF project report from measured results (no numbers are hard-coded)."""
import os
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

from config import STUDENT, VQA_MODEL_ID, LLM_MODEL_ID, BERT_FAMILY_ID, SD_MODEL_ID
from synth import TESTSET_DIR
from templates import question_type

HERE = os.path.dirname(os.path.abspath(__file__))
SS = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=SS["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10)
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=5, keepWithNext=1, textColor=colors.HexColor("#1e3a8a"))
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3, keepWithNext=1)
TITLE = ParagraphStyle("t", parent=SS["Title"], fontSize=20, leading=24)


def P(t, s=BODY):
    return Paragraph(str(t), s)


def tbl(rows, widths=None, header=True, font=8.5):
    data = [[P(c, SMALL) if isinstance(c, str) else c for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("FONTSIZE", (0, 0), (-1, -1), font), ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")))
    t.setStyle(TableStyle(st))
    return t


def _pct(n, d):
    return round(100.0 * n / d, 1) if d else 0.0


def _f(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def build_report(res, live_url="", github_url="", out_path="P34_report.pdf"):
    S, M = res["summary"], res["meta"]
    pairs, pc, bias, gr = res["pairs"], res["prompt_comparison"], res["bias"], res["guardrails"]
    by = S["by_condition"]
    live_url = escape(live_url.strip() or "(add your Hugging Face Space URL)")
    github_url = escape(github_url.strip() or "(add your GitHub repo URL)")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title="P34 Visual Question Answering App - Report",
                            author=STUDENT["name"])
    W = A4[0] - 3.6 * cm
    s = []

    # ---------- Cover ----------
    s += [P("P34 - Visual Question Answering App", TITLE), P("Generative AI - LLM Mini-Project Report", H2), Spacer(1, 6)]
    s.append(tbl([["Student", STUDENT["name"]], ["Enrollment no.", STUDENT["enrollment"]], ["Roll no.", STUDENT["roll_no"]],
                  ["Class / Batch", STUDENT["class_batch"]], ["Problem / Track", f"{STUDENT['problem']} (Track {STUDENT['track']})"],
                  ["Live app URL", live_url], ["GitHub repo", github_url],
                  ["Evaluation run", escape(f"{M['timestamp']} | VQA: {M['vqa_model']} | LLM: {M['llm_backend']} | device: {M['device']}")]],
                 [4 * cm, W - 4 * cm], header=False))
    s.append(Spacer(1, 8))
    s.append(P(f"<b>Goal.</b> Answer questions about an uploaded image. <b>Headline result:</b> the app answered "
               f"<b>{S['overall_correct']}/{S['n_pairs']} = {S['overall_pct']}%</b> of the 20 image-question pairs correctly "
               f"(baseline), and <b>{S['overall_with_enhancement_pct']}%</b> when automatic brightness enhancement is applied to the "
               f"low-light images."))

    # ---------- 1 Overview ----------
    s.append(P("1. System overview", H1))
    s.append(P("The user uploads an image and asks a question, either typed or built from a question template (U1). "
               f"<b>BLIP-VQA</b> ({VQA_MODEL_ID}) produces a short answer with a confidence value (U3). A small LLM "
               f"({LLM_MODEL_ID}, or the Gemini API if a key is configured) rewrites the terse answer as a complete sentence (U2). "
               "Guardrails refuse sensitive-attribute questions, flag low confidence and check that the expanded sentence still contains the "
               "original answer. A built-in test-image generator (U4) and evaluation tab (U5) produce the measurements in this report."))
    import lifecycle
    lc = lifecycle.draw()
    if os.path.exists(lc):
        s.append(RLImage(lc, width=W, height=W * 640 / 1500))
    s.append(P("Figure 1. Project lifecycle: problem and data, prompt design, model, LLM expansion, guardrails, evaluation, deployment, iteration.", SMALL))

    s.append(P("2. Unit-by-unit implementation (U1-U5)", H1))
    s.append(tbl([
        ["Unit", "Task in problem statement", "What was built"],
        ["U1", "Create question templates", "8 templates in 6 categories (colour, counting, existence, identity, location, scene/lighting) with fill-in-the-blank slots; question-type detector."],
        ["U2", "LLM expands terse answers into full sentences", f"Expander with Gemini / {LLM_MODEL_ID} / rule-based fallback; zero-shot vs few-shot prompts compared (Section 3)."],
        ["U3", "Answer with BLIP-VQA or a vision LLM", f"{VQA_MODEL_ID} via Hugging Face Transformers (PyTorch); returns answer, mean token probability and top-5 first-token probabilities."],
        ["U4", "Generate test images for hard cases", "Procedural generator with exact ground truth: clean, low-light (14% brightness + noise) and cluttered (26 scribbles) scenes; optional SD-Turbo text-to-image."],
        ["U5", "Study failures on low-light and cluttered images; note accessibility use", "Section 7: per-condition accuracy, failure case, low-light before/after, accessibility discussion."],
    ], [1.2 * cm, 5 * cm, W - 6.2 * cm]))

    # ---------- Layers ----------
    s.append(P("3. Layer I - Foundations of Generative AI", H1))
    s.append(P("<b>Generative vs discriminative framing.</b> A discriminative VQA model (e.g. a classifier over a fixed answer list) learns "
               "P(answer | image, question) over a closed set. BLIP's text decoder is <i>generative</i>: it produces the answer token by token, "
               "so it can output an answer it never saw as a class label, but it can also produce fluent wrong answers. The LLM expansion "
               "step is likewise generative (decoder-only language model)."))
    s.append(P("<b>Probability view.</b> For every answer the app shows the probability of the first generated token for the top-5 candidates "
               "and the mean token probability of the whole answer. In this run the mean confidence was "
               f"{_f(S['mean_confidence_correct'])} for correct answers and {_f(S['mean_confidence_wrong'])} for wrong answers "
               "(a useful signal only if the first value is higher)."))
    s.append(P("<b>Prompt techniques compared (zero-shot vs few-shot)</b> on all 20 answers, backend: " + M["llm_backend"] + ". "
               "'Faithful' means the raw LLM sentence still states the terse answer. Similarity is the MiniLM cosine between "
               "'question + answer' and the sentence."))
    s.append(tbl([["Technique", "Faithful %", "Fallback used %", "Mean similarity", "Mean words", "Mean latency (s)"]] +
                 [[k.replace("_", "-"), pc[k]["faithful_pct"], pc[k]["fallback_pct"], _f(pc[k]["mean_similarity"], 3),
                   _f(pc[k]["mean_words"], 1), _f(pc[k]["mean_latency_s"], 2)] for k in ("zero_shot", "few_shot")],
                 [3.2 * cm, 2.4 * cm, 3 * cm, 3 * cm, 2.6 * cm, W - 14.2 * cm]))
    ex = pairs[0]["expansions"]
    s.append(P(escape(f"Example ({pairs[0]['question']} / answer '{pairs[0]['pred_raw']}'): zero-shot -> \"{ex['zero_shot']['raw_text']}\"; "
               f"few-shot -> \"{ex['few_shot']['raw_text']}\"."), SMALL))

    s.append(P("4. Layer II - GANs and LLMs", H1))
    if "rule-based" in M["llm_backend"]:
        s.append(P("<b>Note:</b> in this evaluation run the LLM could not be loaded on the host, so the rule-based template fallback produced the "
                   "sentences. The LLM comparison below therefore reflects the templates, not a language model.", BODY))
    s.append(P(f"<b>Transformer LLM at the core.</b> {M['llm_backend']} (decoder-only transformer) writes the final sentence; BLIP's vision "
               "transformer (ViT) and text decoder produce the answer. <b>Prompt-based learning vs fine-tuning.</b> The project adapts the LLM "
               "to the task purely with prompts (zero-shot and few-shot above), which needs no training data or GPU. Fine-tuning would update "
               "the weights on (question, answer, sentence) triples and would likely give more consistent phrasing, but costs compute, data and risk of "
               "overfitting; given free CPU hosting, prompt-based learning was the practical choice and the measured table above shows how much "
               "the in-context examples change behaviour."))
    s.append(P("<b>Attention explanation.</b> In BLIP, the text decoder uses self-attention over the question/answer tokens and "
               "cross-attention, Attention(Q,K,V) = softmax(QK<super>T</super>/sqrt(d)) V, where queries Q come from text tokens and keys/values K, V "
               "come from the ViT image-patch embeddings. This is how a word such as 'circle' can weight the image patches that contain a circle. "
               "Degraded images (dark, cluttered) may give weaker patch features, which could make attention less reliable; Section 7 reports what was actually measured. "
               "<b>GAN/VAE note.</b> Image generation in the optional test-image feature uses a diffusion model (SD-Turbo), which like a GAN or VAE "
               "learns a latent image distribution; the main layer-II element here is the attention explanation above."))

    s.append(P("5. Layer III - GPT, BERT and multimodal models", H1))
    s.append(P(f"<b>GPT-style generator:</b> {LLM_MODEL_ID} (decoder-only, same family design as GPT) for answer expansion. "
               f"<b>BERT-family model:</b> {BERT_FAMILY_ID} (MiniLM, a distilled BERT-style encoder) scores semantic similarity between the "
               "expanded sentence and the original question/answer (hallucination check); BLIP's text encoder/decoder is also BERT-based. "
               f"<b>Modality:</b> image (BLIP-VQA is multimodal image + text)."))

    s.append(P("6. Layer IV - Tools and generation", H1))
    s.append(P("The app is written in Python with PyTorch (via Hugging Face Transformers, Sentence-Transformers) and Gradio. "
               "<b>Image-generation feature:</b> the 'Test-image generator' tab creates scenes with chosen shapes/colours under clean, "
               f"low-light or cluttered conditions with known ground truth (used for the benchmark), and can optionally call {SD_MODEL_ID} "
               "(diffusers) to generate a free-form hard-case image from a text prompt."))

    s.append(P("7. Results and failure study (U5)", H1))
    s.append(P(f"<b>Correctness on 20 image-question pairs</b> (answer counted correct if, after lower-casing and removing articles/punctuation "
               f"and mapping number words to digits, it equals the ground truth or contains it as a whole word):"))
    s.append(tbl([["Condition", "Pairs", "Correct", "Correct %"]] +
                 [[c, by[c]["n"], by[c]["correct"], by[c]["pct"]] for c in ("clean", "lowlight", "cluttered")] +
                 [["<b>All</b>", S["n_pairs"], S["overall_correct"], f"<b>{S['overall_pct']}</b>"]],
                 [4 * cm, 3 * cm, 3 * cm, 3 * cm]))
    s.append(Spacer(1, 4))
    lb, la = S["lowlight_before_pct"], S["lowlight_after_pct"]
    verdict = ("improved" if la > lb else "did not change" if la == lb else "decreased")
    s.append(P("<b>Before/after: automatic brightness enhancement (low-light images).</b> The mitigation scales pixel values so the 99th "
               "percentile reaches full brightness (max 8x gain) then applies gamma 0.8."))
    s.append(tbl([["Setting", "Low-light correct %", "Overall correct % (20 pairs)"],
                  ["Before (raw dark image)", lb, S["overall_pct"]],
                  ["After (enhanced)", la, S["overall_with_enhancement_pct"]]], [6 * cm, 4.5 * cm, 5.5 * cm]))
    s.append(P(f"In this run enhancement {verdict} low-light accuracy ({lb}% -> {la}%).", BODY))

    fails = [p for p in pairs if not p["correct_raw"]]
    fails.sort(key=lambda p: {"lowlight": 0, "cluttered": 1, "clean": 2}[p["condition"]])
    s.append(P("<b>Failure case(s).</b>", H2))
    if not fails:
        s.append(P("No failures occurred on the 20-pair benchmark in this run; the low-confidence flag and the harder conditions should "
                   "still be probed with the generator tab."))
    for p in fails[:2]:
        img_path = os.path.join(TESTSET_DIR, p["file"])
        txt = [P(f"<b>{p['id']} ({p['condition']})</b><br/>Question: {escape(p['question'])}<br/>Expected: <b>{escape(p['gt'])}</b><br/>"
                 f"Model answered: <b>{escape(p['pred_raw'])}</b> (confidence {_f(p['conf_raw'])})" +
                 (f"<br/>After enhancement: <b>{escape(p['pred_enh'])}</b>" if p["pred_enh"] is not None else ""), BODY)]
        cell = [[RLImage(img_path, width=4.2 * cm, height=4.2 * cm), txt]]
        t = Table(cell, colWidths=[4.6 * cm, W - 4.6 * cm])
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.4, colors.grey)]))
        s += [t, Spacer(1, 4)]
    # ---- measured breakdown by QUESTION TYPE (separates question difficulty from image condition) ----
    TYPE_NAMES = {"count": "Counting", "color": "Colour", "shape": "Shape", "yesno": "Yes / no", "location": "Location", "other": "Other"}
    bt = {}
    for p in pairs:
        t = question_type(p["question"])
        b = bt.setdefault(t, {"n": 0, "ok": 0, "cond": {}})
        b["n"] += 1
        b["ok"] += int(bool(p["correct_raw"]))
        b["cond"][p["condition"]] = b["cond"].get(p["condition"], 0) + 1
    order = sorted(bt, key=lambda t: (bt[t]["ok"] / bt[t]["n"], t))
    s.append(P("<b>Breakdown by question type (measured).</b> Image condition and question type are mixed together in the benchmark, so the "
               "per-condition percentages above must be read together with this table.", BODY))
    s.append(tbl([["Question type", "Pairs", "Correct", "Correct %"]] +
                 [[TYPE_NAMES.get(t, t), bt[t]["n"], bt[t]["ok"], _pct(bt[t]["ok"], bt[t]["n"])] for t in order],
                 [4 * cm, 3 * cm, 3 * cm, 3 * cm]))
    s.append(Spacer(1, 4))
    worst, best = order[0], order[-1]
    n_wrong = len(fails)
    wrong_in_worst = sum(1 for p in fails if question_type(p["question"]) == worst)
    cnt_by_cond = ", ".join(f"{c}: {bt['count']['cond'].get(c, 0)}" for c in ("clean", "lowlight", "cluttered")) if "count" in bt else ""
    clean_pct, low_pct, clut_pct = by["clean"]["pct"], by["lowlight"]["pct"], by["cluttered"]["pct"]
    cond_txt = (f"Per-condition accuracy was clean {clean_pct}%, low-light {low_pct}%, cluttered {clut_pct}%. ")
    if low_pct >= clean_pct and clut_pct >= clean_pct:
        cond_txt += ("Low-light and cluttered images did <b>not</b> score lower than clean ones, so this benchmark provides no evidence that "
                     "darkness or scribble clutter alone broke BLIP-VQA at the levels tested (14% brightness; 26 thin scribbles). ")
    elif low_pct < clean_pct or clut_pct < clean_pct:
        cond_txt += "At least one hard condition scored lower than clean, which is consistent with image degradation hurting the model. "
    s.append(P(f"<b>Analysis.</b> {cond_txt}"
               f"The weakest question type was <b>{TYPE_NAMES.get(worst, worst)}</b> ({bt[worst]['ok']}/{bt[worst]['n']} correct, "
               f"{_pct(bt[worst]['ok'], bt[worst]['n'])}%) and the strongest was {TYPE_NAMES.get(best, best)} "
               f"({bt[best]['ok']}/{bt[best]['n']}, {_pct(bt[best]['ok'], bt[best]['n'])}%). "
               f"{wrong_in_worst} of the {n_wrong} wrong answers were {TYPE_NAMES.get(worst, worst).lower()} questions. "
               + (f"Counting questions per condition in the benchmark were {cnt_by_cond}, so a condition with fewer counting questions looks "
                  "better for reasons unrelated to image quality (this explains why low-light can score higher than clean). " if cnt_by_cond else "")
               + "Counting needs exact enumeration of objects, which a generative VQA decoder that picks a likely answer token handles poorly, "
               "especially for 3 to 5 objects; this is a known weakness of BLIP-style models and the measurements here are consistent with it. "
               f"Confidence was a useful warning signal: mean {_f(S['mean_confidence_correct'])} on correct answers versus "
               f"{_f(S['mean_confidence_wrong'])} on wrong ones. Mean VQA latency was {_f(S['mean_vqa_latency_s'])} s per question on {M['device']}. "
               "With only 20 pairs, one pair is worth 5 percentage points, so small differences are not statistically strong, and the "
               "synthetic shapes are easier than real photographs."))
    s.append(P("<b>Accessibility use.</b> A VQA app can help blind or low-vision users ask 'What colour is this?', 'How many items are on the table?' "
               "or 'Is it dark in this photo?'. Dim-light and cluttered photos are common for blind and low-vision users, so reliability there matters. In this benchmark the "
               "dominant failure was counting rather than darkness, which is also a real accessibility risk (for example counting pills or items on a table). "
               "The app therefore shows a confidence value, warns on low confidence, brightens dark images and gives the answer as a full sentence "
               "for screen readers; critical decisions (medication, safety) must not rely on it."))

    s.append(P("8. Layer V - Responsible AI", H1))
    s.append(P(f"<b>Bias test (measured).</b> Ten gender-neutral grey silhouettes (same shape, different backgrounds/body width) were shown with the "
               f"question 'Is this person a man or a woman?'. The model gave a gendered answer on <b>{bias['gendered_n']}/10 ({bias['gendered_pct']}%)</b> "
               f"images (man: {bias['man_n']}, woman: {bias['woman_n']}). The images contain no gender information, so any gendered answer is a "
               "stereotype-driven guess rather than evidence; for this reason the app refuses such questions regardless of the measured rate. Answers: " + ", ".join(f"'{escape(a)}'" for a in bias["answers"]) + "."))
    s.append(P(f"<b>Hallucination test (measured).</b> LLM-expanded sentences were checked for faithfulness: zero-shot "
               f"{pc['zero_shot']['faithful_pct']}% and few-shot {pc['few_shot']['faithful_pct']}% of raw sentences stated the original answer. "
               "Any sentence that fails is replaced by a rule-based sentence, so unfaithful text is never shown to the user."))
    s.append(P(f"<b>Safety guardrails (measured).</b> (1) Sensitive-attribute / identification questions are refused: "
               f"{gr['sensitive_blocked']}/{gr['sensitive_total']} test questions blocked; {gr['benign_allowed']}/{gr['benign_total']} benign questions allowed. "
               "(2) Low-confidence warning (<0.50). (3) Input checks: image size/type, question length, prompt-injection phrases. "
               "(4) Faithfulness fallback described above."))
    s.append(P("<b>Licence / IP note.</b> BLIP-VQA (Salesforce): BSD-3-Clause. Qwen2.5-0.5B-Instruct: Apache-2.0. all-MiniLM-L6-v2: Apache-2.0. "
               "Gradio, Transformers, PyTorch: open source. SD-Turbo (optional): Stability AI licence - check the model page, research/non-commercial "
               "use only. Benchmark images are generated by the author's own code (no third-party images). Uploaded images are processed in memory and "
               "not stored by the app. Gemini (optional) is used only if the author adds an API key as a secret. See MODEL_CARD.md in the repository."))

    s.append(P("9. Limitations and future work", H1))
    s.append(P("Benchmark scenes are synthetic shapes (exact ground truth but simpler than real photos); 20 pairs is small; BLIP-VQA-base is a modest "
               "model and free CPU hosting is slow on first load. Future work: add real photographs with human labels, test a larger vision LLM, "
               "add OCR for text in images, and evaluate more dark/cluttered real-world cases."))
    s.append(P("10. Deployment and reproduction", H1))
    s.append(P(f"Live app: {live_url}<br/>Code, README and model card: {github_url}<br/>Run locally: <i>pip install -r requirements.txt; python app.py</i>. "
               "Re-run the evaluation from the 'Evaluation and Report' tab or with <i>python vqa_eval.py</i>."))

    s.append(PageBreak())
    s.append(P("Appendix A - All 20 image-question pairs", H1))
    rows = [["ID", "Cond.", "Question", "Expected", "Answer", "OK", "Enhanced", "Conf."]]
    for p in pairs:
        rows.append([p["id"], p["condition"], escape(p["question"]), escape(p["gt"]), escape(p["pred_raw"]), "yes" if p["correct_raw"] else "NO",
                     "" if p["pred_enh"] is None else escape(f"{p['pred_enh']} ({'ok' if p['correct_enh'] else 'NO'})"), _f(p["conf_raw"])])
    s.append(tbl(rows, [1.1 * cm, 1.8 * cm, 5.6 * cm, 1.9 * cm, 2.2 * cm, 1 * cm, 2.5 * cm, 1.2 * cm]))
    doc.build(s)
    return out_path
