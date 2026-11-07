"""Fast pre-flight check: proves the REAL models load and answer before you spend
20 minutes on the full evaluation.  Run:  python smoke_test.py
Expected: 4 lines of OK.  Anything else -> fix that before running vqa_eval.py."""
import sys, time, traceback

def step(n, what, fn):
    t0 = time.time()
    try:
        out = fn()
    except Exception:
        print(f"[{n}] FAIL  {what}\n{traceback.format_exc()}")
        return None
    print(f"[{n}] OK    {what}  ({time.time() - t0:.1f}s)  -> {out}")
    return out

def main():
    import synth
    from vqa_model import VQAModel
    from llm import Expander
    from bert_check import SimilarityScorer
    from config import MOCK
    if MOCK:
        print("ERROR: VQA_MOCK is set. Unset it - smoke test must use the real models.")
        return 1

    img = synth.render([("circle", "red"), ("square", "blue")], "clean", seed=1)
    vqa, exp, sc = VQAModel(), Expander(), SimilarityScorer()

    r = step(1, "BLIP-VQA loads and answers", lambda: vqa.answer(img, "What color is the circle?"))
    if r is None:
        return 1
    print(f"      answer={r['answer']!r} confidence={r['confidence']} top5={r['top5']}")
    if r["confidence"] is None:
        print("      WARN: confidence is None -> probability view will show n/a (see vqa_model.py).")
    if not r["top5"]:
        print("      WARN: top5 empty -> the Layer-I probability view will be blank.")

    e = step(2, f"LLM expansion backend = {exp.name}", lambda: exp.expand("What color is the circle?", r["answer"], "few_shot"))
    if e is not None:
        print(f"      backend={e['backend']} raw={e['raw_text']!r} fallback_used={e['fallback_used']}")
        if e["backend"] == "template":
            print("      WARN: the LLM did not load, so sentences come from rule-based templates.")
            print("            The report will say so. Check RAM / the traceback above.")

    step(3, "BERT-family similarity (MiniLM)", lambda: round(sc.similarity("a red circle", "The circle is red."), 3))
    step(4, "Benchmark images present", lambda: f"{len(synth.load_testset())} pairs in {synth.TESTSET_DIR}")
    print("\nSmoke test finished. If steps 1 and 2 are OK, run:  python vqa_eval.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())
