"""Layer III - BERT-family model: MiniLM sentence encoder (distilled from BERT-style transformers).
Used to measure semantic faithfulness of LLM-expanded answers (hallucination check)."""
from config import BERT_FAMILY_ID, MOCK


class SimilarityScorer:
    def __init__(self):
        self.mock = MOCK
        self._m = None

    @property
    def name(self):
        return "MOCK" if self.mock else BERT_FAMILY_ID

    def load(self):
        if self.mock or self._m is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(BERT_FAMILY_ID, device="cpu")

    def similarity(self, a: str, b: str):
        """Cosine similarity in [-1, 1] (None in mock mode)."""
        if self.mock:
            return None
        self.load()
        from sentence_transformers import util
        ea, eb = self._m.encode([a, b], convert_to_tensor=True)
        return float(util.cos_sim(ea, eb))
