# Assignment requirements (verbatim transcription of the 3 screenshots – source of truth)

## Page "P34" (problem statement)
**P34 · Visual Question Answering App**
**Goal:** Answer questions about an uploaded image.
1. U1 — Create question templates.
2. U2 — Have the LLM expand terse answers into full sentences.
3. U3 — Answer with BLIP-VQA or a vision LLM.
4. U4 — Generate test images for hard cases.
5. U5 — Study failures on low-light and cluttered images; note accessibility use.
**Deliverables:** VQA app; correctness % on 20 image–question pairs.

## Page "Generative AI – 60 LLM Mini-Project Problem Statements"
Key points (CA assessment)
- Allocation: Roll No. 1 → P1, Roll No. 2 → P2 … Roll No. 60 → P60. No swaps.  (Roll 34 → P34)
- Individual work only — no groups.
- Complete all 5 tasks (U1–U5) of your problem and **deploy it as a live app**.
- Submit in the class shared drive, folder **RollNo_Name**: PDF report, live app URL + GitHub link, 2–3 min demo video.
- Deadline: **Saturday, 7 November 2026**. Late submission reduces marks.

Overview: Sixty LLM-centred mini-projects, each covering all five syllabus units, so every student works through the whole of Generative AI in one deployable app. Each project has a focus area (Tracks A–E) and five required layers, one per unit, listed in its row. None repeat the remedial list R1–R20.

Rules for every project
- One core feature working end to end; extras only after that works.
- Must be deployed with a public URL (Hugging Face Spaces, Streamlit Community Cloud or Render) and demonstrated live.
- API keys go in the platform's secrets settings, never in the code or repo.
- Each project reports at least one measurable result (accuracy, score, latency or a before/after table) and one failure case.
- Allocation by roll number; no swaps; each student works individually; group work not allowed.

How to read each problem statement: a one-line goal, five tasks (U1–U5), one per syllabus unit, done in order, and the project-specific deliverables. Every project also submits the **standard deliverables: live app URL, public GitHub repo with README and model card, PDF report, and a 2–3 minute demo video.**

Common free stack (suggestions): LLM – free-tier API (Gemini, Groq) or small open model from Hugging Face (Qwen2.5-0.5B/1.5B-Instruct, TinyLlama, Phi). Transformers/training – HF transformers, datasets, peft, PyTorch or TensorFlow 2/Keras. Images/audio – diffusers (sd-turbo, InstructPix2Pix), BLIP, CLIP, Whisper, MusicGen, gTTS/SpeechT5. Retrieval – sentence-transformers, FAISS or Chroma. UI + hosting – Gradio or Streamlit on HF Spaces / Streamlit Community Cloud. Heavy image or music models may call a hosted inference API instead of running on the free CPU host; a Colab + Gradio share link is acceptable only as a backup on demo day.

**Five layers every project must include**
| Layer | Unit | Minimum requirement in every project |
|---|---|---|
| I Foundations | Introduction to Generative AI | Lifecycle diagram; generative vs discriminative framing; at least 2 prompt techniques compared; one probability or latent-space view |
| II GANs & LLMs | Introduction to GANs & LLM | Transformer LLM at the core; fine-tuning or prompt-based learning compared; one GAN/VAE element or attention explanation |
| III GPT, BERT & Multimodal | Generative AI Models | A GPT-style generator plus a BERT-family model; at least one image or audio modality |
| IV Tools & Generation | Generative AI Tools & Techniques | Built in Python with PyTorch or TensorFlow 2; at least one image, video or music generation feature |
| V Responsible AI | Ethical, Legal, Practical | One measured test of bias or hallucination; a safety guardrail; licence/IP note; model card in the repo |

Tracks table (partially visible): A P1–P12 Foundations & prompt engineering; B P13–P24 GANs & transformer internals; C P25–P36 GPT, BERT and multimodal models (P34 is in Track C); all CO1–CO5. (Rows D/E were cut off in the screenshot.)

## Student
Saumya Joshi · Enrollment ADT23SOCB0997 · Roll 34 · AIA-05 Batch B → drive folder name: `34_Saumya_Joshi`
