"""P34 - Visual Question Answering App (Gradio).  Run: python app.py"""
import os
import tempfile

import gradio as gr

import synth
import templates as T
from bert_check import SimilarityScorer
from config import STUDENT, MOCK, DARK_MEAN_THRESHOLD
from vqa_eval import run_evaluation, save_results, summary_markdown
from guardrails import check_question, prepare_image, confidence_flag
from llm import Expander
from report import build_report
from vqa_model import VQAModel

VQA, LLM, SCORER = VQAModel(), Expander(), SimilarityScorer()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(tempfile.gettempdir(), "p34_outputs")


# ---------------- Tab 1: Ask ----------------
def build_q(label, slot):
    try:
        return T.build_question(label, slot), ""
    except ValueError as e:
        return gr.update(), str(e)


def ask(image, question, brighten, expand, technique):
    img, msg = prepare_image(image)
    if img is None:
        return msg, "", "", {}, ""
    ok, msg = check_question(question)
    if not ok:
        return msg, "", "", {}, ""
    notes = []
    if brighten and synth.mean_luminance(img) < DARK_MEAN_THRESHOLD:
        img = synth.enhance_low_light(img)
        notes.append("The image was dark, so it was automatically brightened before answering.")
    r = VQA.answer(img, question.strip())
    ans = r["answer"]
    conf = r["confidence"]
    flag = confidence_flag(conf)
    if flag:
        notes.append(flag)
    sentence = ""
    if expand:
        e = LLM.expand(question.strip(), ans, technique)
        sentence = e["text"]
        if e["fallback_used"]:
            notes.append("The LLM sentence did not state the model's answer, so a safe template sentence was used instead.")
    conf_txt = "n/a" if conf is None else f"{conf:.2f}"
    banner = "MOCK MODE (tests only) - " if MOCK else ""
    return (f"**Answer:** {ans}", sentence, f"{banner}confidence {conf_txt} | {r['latency']:.1f}s | VQA: {VQA.name} | LLM: {LLM.name}",
            r["top5"], "\n".join(notes))


# ---------------- Tab 2: Generator (U4) ----------------
def gen_scene(shapes_text, condition, seed):
    try:
        shapes = synth.parse_shapes(shapes_text)
        img = synth.render(shapes, condition, seed=int(seed))
    except ValueError as e:
        return None, str(e)
    qa = synth.qa_from_shapes(shapes)
    info = "**Ground truth questions for this image:**\n" + "\n".join(f"- {q}  ->  **{a}**" for q, a in qa)
    return img, info


def gen_sd(prompt):
    if not prompt.strip():
        return None, "Type a prompt first."
    try:
        return synth.generate_with_sd_turbo(prompt.strip()), "Generated with SD-Turbo (educational use; see model card for licence)."
    except Exception as e:
        return None, f"SD-Turbo is not available on this host ({type(e).__name__}). The procedural generator above still works."


# ---------------- Tab 3: Evaluation + report ----------------
def run_eval_ui(live_url, gh_url):
    os.makedirs(OUT_DIR, exist_ok=True)
    log, res = [], None
    for kind, payload in run_evaluation(VQA, LLM, SCORER):
        if kind == "progress":
            log.append(payload)
            yield "\n".join(log[-8:]), None, "", None, None, None
        else:
            res = payload
    json_path = save_results(res, os.path.join(OUT_DIR, "results.json"))
    md_path = os.path.join(OUT_DIR, "results_summary.md")
    with open(md_path, "w") as fh:
        fh.write(summary_markdown(res))
    pdf_path = build_report(res, live_url, gh_url, os.path.join(OUT_DIR, "P34_report_Saumya_Joshi.pdf"))
    S, pc = res["summary"], res["prompt_comparison"]
    md = (f"### Results\n- **Correctness on 20 pairs: {S['overall_correct']}/{S['n_pairs']} = {S['overall_pct']}%**\n"
          f"- clean {S['by_condition']['clean']['pct']}% | low-light {S['by_condition']['lowlight']['pct']}% | cluttered {S['by_condition']['cluttered']['pct']}%\n"
          f"- low-light after brightening: {S['lowlight_after_pct']}% (overall {S['overall_with_enhancement_pct']}%)\n"
          f"- prompt faithfulness: zero-shot {pc['zero_shot']['faithful_pct']}% vs few-shot {pc['few_shot']['faithful_pct']}%\n"
          f"- bias test: gendered answers on neutral silhouettes {res['bias']['gendered_pct']}%\n"
          f"- guardrails: {res['guardrails']['sensitive_blocked']}/{res['guardrails']['sensitive_total']} sensitive blocked, "
          f"{res['guardrails']['benign_allowed']}/{res['guardrails']['benign_total']} benign allowed\n\n"
          + ("**MOCK MODE: these numbers are NOT real.**" if res["meta"]["mock"] else "Download the PDF report and results.json below."))
    rows = [[p["id"], p["condition"], p["question"], p["gt"], p["pred_raw"], "yes" if p["correct_raw"] else "NO",
             "" if p["pred_enh"] is None else p["pred_enh"]] for p in res["pairs"]]
    yield "Done.", rows, md, pdf_path, json_path, md_path


def about_text():
    p = os.path.join(HERE, "MODEL_CARD.md")
    return open(p).read() if os.path.exists(p) else "Model card not found."


with gr.Blocks(title="P34 Visual Question Answering App") as demo:
    gr.Markdown(f"# P34 - Visual Question Answering App\n{STUDENT['name']} | {STUDENT['enrollment']} | Roll {STUDENT['roll_no']} | {STUDENT['class_batch']}  \n"
                "Upload an image, ask a question (or build one from a template). BLIP-VQA answers; an LLM turns the short answer into a sentence. "
                "The first request loads the models and can take 1-2 minutes.")
    if MOCK:
        gr.Markdown("**MOCK MODE is ON (tests only).**")
    with gr.Tabs():
        with gr.Tab("Ask a question"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(type="pil", label="Upload an image", height=320)
                    tpl = gr.Dropdown(T.template_labels(), value=T.template_labels()[0], label="U1 - Question template")
                    slot = gr.Textbox(label="Fill the blank (e.g. circle, squares, blue)", placeholder="circle")
                    build_btn = gr.Button("Build question from template")
                    q_in = gr.Textbox(label="Your question", placeholder="What color is the circle?")
                    with gr.Row():
                        brighten = gr.Checkbox(True, label="Auto-brighten dark images")
                        expand = gr.Checkbox(True, label="Expand answer into a sentence (LLM)")
                    technique = gr.Dropdown([("Few-shot prompt", "few_shot"), ("Zero-shot prompt", "zero_shot")],
                                            value="few_shot", label="LLM prompt technique")
                    ask_btn = gr.Button("Ask", variant="primary")
                with gr.Column():
                    ans_out = gr.Markdown(label="Answer")
                    sent_out = gr.Textbox(label="Full-sentence answer (LLM)", interactive=False)
                    meta_out = gr.Textbox(label="Details", interactive=False)
                    prob_out = gr.Label(label="Probability view: top candidates for the first answer token", num_top_classes=5)
                    warn_out = gr.Textbox(label="Notes and warnings", interactive=False)
            build_btn.click(build_q, [tpl, slot], [q_in, warn_out])
            ask_btn.click(ask, [img_in, q_in, brighten, expand, technique], [ans_out, sent_out, meta_out, prob_out, warn_out])
        with gr.Tab("Test-image generator (U4)"):
            gr.Markdown("Create hard-case images with **known ground truth**. Format: `shape:color` (circle/square/triangle; red/blue/green/yellow), up to 4.")
            with gr.Row():
                with gr.Column():
                    shapes_in = gr.Textbox("circle:red, square:blue", label="Shapes")
                    cond_in = gr.Dropdown(["clean", "lowlight", "cluttered"], value="lowlight", label="Condition")
                    seed_in = gr.Number(7, label="Seed", precision=0)
                    gen_btn = gr.Button("Generate image", variant="primary")
                    use_btn = gr.Button("Send image to the 'Ask' tab")
                with gr.Column():
                    gen_img = gr.Image(type="pil", label="Generated image", height=320)
                    gen_info = gr.Markdown()
            gen_btn.click(gen_scene, [shapes_in, cond_in, seed_in], [gen_img, gen_info])
            use_btn.click(lambda x: x, gen_img, img_in)
            gr.Markdown("**Optional:** free-form generation with a diffusion model (SD-Turbo). Slow on free CPU; may be unavailable.")
            sd_prompt = gr.Textbox(label="Prompt", placeholder="a dark cluttered desk at night with a red mug")
            sd_btn = gr.Button("Generate with SD-Turbo")
            sd_img = gr.Image(type="pil", label="SD-Turbo image", height=320)
            sd_msg = gr.Markdown()
            sd_btn.click(gen_sd, sd_prompt, [sd_img, sd_msg])
        with gr.Tab("Evaluation and Report (U5)"):
            gr.Markdown("Runs the fixed **20 image-question pairs** (clean / low-light / cluttered), the low-light before/after test, the prompt comparison, "
                        "the bias test and the guardrail test, then builds the **PDF report**. Takes a few minutes on free CPU.")
            live_in = gr.Textbox(label="Live app URL (for the report)", placeholder="https://huggingface.co/spaces/<user>/<space>")
            gh_in = gr.Textbox(label="GitHub repo URL (for the report)", placeholder="https://github.com/<user>/<repo>")
            ev_btn = gr.Button("Run evaluation and build report", variant="primary")
            ev_status = gr.Textbox(label="Progress", lines=8, interactive=False)
            ev_md = gr.Markdown()
            ev_tbl = gr.Dataframe(headers=["id", "condition", "question", "expected", "answer", "correct", "after brighten"], label="Per-pair results")
            ev_pdf = gr.File(label="PDF report")
            ev_json = gr.File(label="results.json")
            ev_sum = gr.File(label="results_summary.md (upload to GitHub)")
            ev_btn.click(run_eval_ui, [live_in, gh_in], [ev_status, ev_tbl, ev_md, ev_pdf, ev_json, ev_sum])
        with gr.Tab("About / Model card"):
            gr.Markdown(about_text())

if __name__ == "__main__":
    demo.launch()
