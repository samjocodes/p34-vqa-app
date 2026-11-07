"""Draws the project lifecycle diagram (PNG) with PIL."""
import os
from PIL import Image, ImageDraw, ImageFont

import tempfile
OUT = os.path.join(tempfile.gettempdir(), "p34_lifecycle.png")


def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw(path=OUT):
    W, H = 1500, 640
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f, fb, fs = _font(24), _font(28), _font(20)
    d.text((W // 2, 30), "P34 VQA app - generative AI lifecycle", font=fb, fill="black", anchor="mm")
    row1 = [("1. Problem & data", "Questions about an\nimage; build 20-pair\nbenchmark (U4)", "#dbeafe"),
            ("2. Prompt design", "Question templates\n(U1) + zero/few-shot\nprompts (U2)", "#dcfce7"),
            ("3. Model", "BLIP-VQA answers\n(U3, ViT + BERT-style\ntransformer)", "#fef9c3"),
            ("4. LLM expansion", "Qwen / Gemini turns\nterse answer into a\nsentence (U2)", "#fce7f3")]
    row2 = [("5. Guardrails", "Sensitive-question\nrefusal, low-confidence\nflag, faithfulness check", "#fee2e2"),
            ("6. Evaluate", "Correctness %, low-light\n& clutter failures,\nbias test (U5)", "#ede9fe"),
            ("7. Deploy", "Gradio on Hugging\nFace Spaces + GitHub\nrepo + model card", "#e0f2fe"),
            ("8. Monitor / iterate", "Failure cases ->\nbetter prompts, brightness\nenhancement, new tests", "#f1f5f9")]

    def boxes(row, y):
        bw, bh, gap, x0 = 320, 200, 50, 40
        pts = []
        for i, (t, s, col) in enumerate(row):
            x = x0 + i * (bw + gap)
            d.rounded_rectangle([x, y, x + bw, y + bh], radius=18, fill=col, outline="#334155", width=3)
            d.text((x + bw // 2, y + 36), t, font=f, fill="black", anchor="mm")
            d.multiline_text((x + bw // 2, y + 120), s, font=fs, fill="#1e293b", anchor="mm", align="center", spacing=6)
            pts.append((x, y, x + bw, y + bh))
        return pts

    p1, p2 = boxes(row1, 80), boxes(row2, 360)

    def arrow(a, b):
        d.line([a, b], fill="#334155", width=4)
        ax, ay = b
        d.polygon([(ax, ay), (ax - 14, ay - 9), (ax - 14, ay + 9)] if b[0] > a[0] else [(ax, ay), (ax + 14, ay - 9), (ax + 14, ay + 9)], fill="#334155")

    for row in (p1, p2):
        for a, b in zip(row, row[1:]):
            arrow((a[2], (a[1] + a[3]) // 2), (b[0], (b[1] + b[3]) // 2))
    d.line([(p1[-1][0] + 160, p1[-1][3]), (p1[-1][0] + 160, 330), (p2[0][0] + 160, 330), (p2[0][0] + 160, p2[0][1] - 4)], fill="#334155", width=4)
    d.polygon([(p2[0][0] + 160, p2[0][1]), (p2[0][0] + 151, p2[0][1] - 14), (p2[0][0] + 169, p2[0][1] - 14)], fill="#334155")
    d.text((W // 2, 610), "Generative (BLIP decoder / LLM) models produce answers token by token; failures feed back into step 2.", font=fs, fill="#475569", anchor="mm")
    img.save(path)
    return path


if __name__ == "__main__":
    print(draw())
