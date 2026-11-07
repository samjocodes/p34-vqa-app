"""U3 - Answer with BLIP-VQA (Salesforce/blip-vqa-base). Returns answer + confidence + top-5 first-token probabilities."""
import time
from config import VQA_MODEL_ID, MOCK


class VQAModel:
    def __init__(self):
        self.mock = MOCK
        self._proc = None
        self._model = None
        self.device = "cpu"

    def load(self):
        if self.mock or self._model is not None:
            return
        import torch
        from transformers import BlipProcessor, BlipForQuestionAnswering
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._proc = BlipProcessor.from_pretrained(VQA_MODEL_ID)
        self._model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL_ID).to(self.device).eval()

    @property
    def name(self):
        return "MOCK (tests only)" if self.mock else VQA_MODEL_ID

    def answer(self, image, question):
        """Returns dict(answer, confidence, top5 {token: prob}, latency)."""
        t0 = time.time()
        if self.mock:
            return {"answer": "red", "confidence": 0.9, "top5": {"red": 0.9, "blue": 0.05},
                    "latency": time.time() - t0}
        import torch
        self.load()
        inputs = self._proc(images=image.convert("RGB"), text=question, return_tensors="pt").to(self.device)
        conf, top5, ans = None, {}, None
        try:
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False,
                                           output_scores=True, return_dict_in_generate=True)
            seq = out.sequences if hasattr(out, "sequences") else out
            ans = self._proc.decode(seq[0], skip_special_tokens=True).strip()
            scores = getattr(out, "scores", None)
            if scores:
                n = len(scores)
                gen = seq[0][-n:]
                special = set(self._proc.tokenizer.all_special_ids)
                probs = []
                for i, tok in enumerate(gen):
                    p = torch.softmax(scores[i][0].float(), dim=-1)
                    if int(tok) not in special:
                        probs.append(float(p[tok]))
                    if i == 0:
                        tp, ti = torch.topk(p, 5)
                        for pv, iv in zip(tp.tolist(), ti.tolist()):
                            label = self._proc.tokenizer.decode([iv]).strip()
                            if label and label not in top5:
                                top5[label] = round(pv, 4)
                conf = sum(probs) / len(probs) if probs else None
        except Exception:
            pass  # keep any answer already decoded; confidence/top5 stay unset
        if not ans:  # robust fallback: plain generation without scores
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=10)
            ans = self._proc.decode(out[0], skip_special_tokens=True).strip()
        return {"answer": ans, "confidence": conf, "top5": top5, "latency": time.time() - t0}
