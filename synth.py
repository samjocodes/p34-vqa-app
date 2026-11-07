"""U4 - Generate test images for hard cases (procedural, ground truth known) + optional SD-Turbo."""
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

SIZE = 384
COLORS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 160, 50), "yellow": (245, 215, 20)}
SHAPES = ("circle", "square", "triangle")
SLOTS = {
    1: [(192, 192)],
    2: [(110, 192), (274, 192)],
    3: [(110, 105), (274, 105), (192, 275)],
    4: [(110, 110), (274, 110), (110, 274), (274, 274)],
}
BG = (235, 235, 235)
CLUTTER_COLORS = [(100, 100, 100), (20, 20, 20), (120, 70, 30), (130, 40, 160)]  # never a target colour

import tempfile
# Prefer the testset committed next to the code (deterministic, same images as the report
# figures); fall back to a temp dir that is built on demand if the repo copy is absent.
_LOCAL_TESTSET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testset")
TESTSET_DIR = (_LOCAL_TESTSET if os.path.exists(os.path.join(_LOCAL_TESTSET, "testset.json"))
               else os.path.join(tempfile.gettempdir(), "p34_testset"))


def _draw_shape(d, shape, color, cx, cy, r=48):
    c = COLORS[color]
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=c)
    elif shape == "triangle":
        d.polygon([(cx, cy - r - 6), (cx - r - 6, cy + r), (cx + r + 6, cy + r)], fill=c)
    else:
        raise ValueError(f"unknown shape {shape}")


def render(shapes, condition="clean", seed=0):
    """shapes: list of (shape, color). condition: clean | lowlight | cluttered."""
    if not 1 <= len(shapes) <= 4:
        raise ValueError("Use 1 to 4 shapes")
    rng = random.Random(seed)
    img = Image.new("RGB", (SIZE, SIZE), BG)
    d = ImageDraw.Draw(img)
    for (shape, color), (cx, cy) in zip(shapes, SLOTS[len(shapes)]):
        _draw_shape(d, shape, color, cx, cy)
    if condition == "cluttered":
        for _ in range(26):  # thin scribbles drawn ON TOP: partial occlusion, answers unchanged
            col = rng.choice(CLUTTER_COLORS)
            x, y = rng.randint(0, SIZE), rng.randint(0, SIZE)
            if rng.random() < 0.5:
                d.line([x, y, x + rng.randint(-70, 70), y + rng.randint(-70, 70)], fill=col, width=3)
            else:
                s = rng.randint(6, 14)
                d.line([x - s, y, x + s, y], fill=col, width=3)
                d.line([x, y - s, x, y + s], fill=col, width=3)
    elif condition == "lowlight":
        arr = np.asarray(img).astype(np.float32) * 0.14
        arr += np.random.RandomState(seed).normal(0, 3.0, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    elif condition != "clean":
        raise ValueError("condition must be clean, lowlight or cluttered")
    return img


def enhance_low_light(img):
    """U5 mitigation: gain so the 99th-percentile pixel reaches ~255 (max 8x), then gamma 0.8."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    p99 = max(float(np.percentile(arr, 99)), 1.0)
    gain = min(255.0 / p99, 8.0)
    arr = np.clip(arr * gain, 0, 255) / 255.0
    arr = np.power(arr, 0.8) * 255.0
    return Image.fromarray(arr.astype(np.uint8))


def mean_luminance(img):
    return float(np.asarray(img.convert("L")).mean())


def qa_from_shapes(shapes):
    """Unambiguous auto-generated (question, answer) pairs for a scene."""
    qa = []
    shape_counts = {s: sum(1 for x, _ in shapes if x == s) for s in SHAPES}
    color_counts = {}
    for _, c in shapes:
        color_counts[c] = color_counts.get(c, 0) + 1
    for s, c in shapes:
        if shape_counts[s] == 1:
            qa.append((f"What color is the {s}?", c))
            break
    for s, c in shapes:
        if color_counts[c] == 1:
            qa.append((f"What shape is the {c} object?", s))
            break
    for s in SHAPES:
        if shape_counts[s] > 0:
            qa.append((f"How many {s}s are there?", str(shape_counts[s])))
    absent = [s for s in SHAPES if shape_counts[s] == 0]
    if absent:
        qa.append((f"Is there a {absent[0]} in the image?", "no"))
    qa.append((f"Is there a {shapes[0][0]} in the image?", "yes"))
    return qa


def parse_shapes(text):
    """'circle:red, square:blue' -> [('circle','red'),('square','blue')]"""
    out = []
    for part in (text or "").split(","):
        part = part.strip().lower()
        if not part:
            continue
        if ":" not in part:
            raise ValueError("Use the form shape:color, e.g. circle:red, square:blue")
        s, c = [x.strip() for x in part.split(":", 1)]
        if s not in SHAPES:
            raise ValueError(f"Shape must be one of {SHAPES}")
        if c not in COLORS:
            raise ValueError(f"Colour must be one of {tuple(COLORS)}")
        out.append((s, c))
    if not 1 <= len(out) <= 4:
        raise ValueError("Give between 1 and 4 shapes")
    return out


# ---- The fixed 20-pair benchmark (ground truth is exact because we draw the scenes) ----
BENCHMARK = [
    # id, condition, shapes, question, answer
    ("c01", "clean", [("circle", "red"), ("square", "blue")], "What color is the circle?", "red"),
    ("c02", "clean", [("triangle", "green"), ("circle", "yellow"), ("square", "red")], "How many shapes are there?", "3"),
    ("c03", "clean", [("square", "blue"), ("square", "blue"), ("square", "green"), ("circle", "red")], "How many squares are there?", "3"),
    ("c04", "clean", [("triangle", "yellow"), ("circle", "blue")], "Is there a triangle in the image?", "yes"),
    ("c05", "clean", [("circle", "green"), ("square", "red")], "Is there a triangle in the image?", "no"),
    ("c06", "clean", [("square", "red"), ("triangle", "blue"), ("circle", "green")], "What shape is the blue object?", "triangle"),
    ("c07", "clean", [("circle", "yellow")], "What color is the circle?", "yellow"),
    ("c08", "clean", [("circle", "red"), ("circle", "green"), ("square", "blue")], "How many circles are there?", "2"),
    ("l01", "lowlight", [("circle", "red"), ("square", "green")], "What color is the circle?", "red"),
    ("l02", "lowlight", [("square", "blue"), ("square", "yellow"), ("triangle", "red")], "How many squares are there?", "2"),
    ("l03", "lowlight", [("triangle", "green"), ("circle", "blue")], "Is there a circle in the image?", "yes"),
    ("l04", "lowlight", [("circle", "yellow"), ("square", "red"), ("triangle", "green")], "What shape is the green object?", "triangle"),
    ("l05", "lowlight", [("square", "red"), ("circle", "blue")], "Is there a triangle in the image?", "no"),
    ("l06", "lowlight", [("triangle", "red")], "What color is the triangle?", "red"),
    ("k01", "cluttered", [("circle", "blue"), ("square", "red")], "What color is the circle?", "blue"),
    ("k02", "cluttered", [("triangle", "yellow"), ("triangle", "green"), ("circle", "red")], "How many triangles are there?", "2"),
    ("k03", "cluttered", [("square", "green"), ("circle", "yellow")], "Is there a square in the image?", "yes"),
    ("k04", "cluttered", [("circle", "red"), ("triangle", "blue"), ("square", "yellow")], "What shape is the yellow object?", "square"),
    ("k05", "cluttered", [("circle", "green"), ("square", "blue")], "Is there a triangle in the image?", "no"),
    ("k06", "cluttered", [("circle", "red"), ("circle", "blue"), ("circle", "green")], "How many circles are there?", "3"),
]


def bias_silhouette(i):
    """Gender-neutral grey silhouette (circle head + rectangle body), varying background/proportions."""
    bgs = [(235, 235, 235), (200, 220, 240), (240, 225, 200), (210, 235, 210), (225, 210, 235)]
    img = Image.new("RGB", (SIZE, SIZE), bgs[i % 5])
    d = ImageDraw.Draw(img)
    body_w = [70, 90][i // 5 % 2]
    g = (90, 90, 90)
    d.ellipse([192 - 38, 70, 192 + 38, 146], fill=g)
    d.rectangle([192 - body_w, 150, 192 + body_w, 270], fill=g)
    d.rectangle([192 - body_w + 10, 270, 192 - 8, 350], fill=g)
    d.rectangle([192 + 8, 270, 192 + body_w - 10, 350], fill=g)
    return img


def build_testset(out_dir=TESTSET_DIR):
    os.makedirs(os.path.join(out_dir, "bias"), exist_ok=True)
    meta = []
    for k, (pid, cond, shapes, q, a) in enumerate(BENCHMARK):
        fn = f"{pid}_{cond}.png"
        render(shapes, cond, seed=100 + k).save(os.path.join(out_dir, fn))
        meta.append({"id": pid, "condition": cond, "file": fn,
                     "shapes": [list(s) for s in shapes], "question": q, "answer": a})
    with open(os.path.join(out_dir, "testset.json"), "w") as f:
        json.dump(meta, f, indent=2)
    for i in range(10):
        bias_silhouette(i).save(os.path.join(out_dir, "bias", f"sil_{i:02d}.png"))
    return meta


def load_testset(out_dir=TESTSET_DIR):
    if not os.path.exists(os.path.join(out_dir, "testset.json")):
        build_testset(out_dir)
    with open(os.path.join(out_dir, "testset.json")) as f:
        return json.load(f)


# ---- Optional: diffusion-model image generation for hard cases (heavy on CPU; lazy) ----
_sd = {"pipe": None}


def generate_with_sd_turbo(prompt, size=384):
    from config import SD_MODEL_ID
    import torch
    from diffusers import AutoPipelineForText2Image
    if _sd["pipe"] is None:
        _sd["pipe"] = AutoPipelineForText2Image.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float32)
        _sd["pipe"].set_progress_bar_config(disable=True)
    out = _sd["pipe"](prompt=prompt, num_inference_steps=1, guidance_scale=0.0, height=size, width=size)
    return out.images[0]


if __name__ == "__main__":
    m = build_testset(_LOCAL_TESTSET)
    print(f"Built {len(m)} benchmark images + 10 bias silhouettes in {_LOCAL_TESTSET}")
