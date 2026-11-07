# START HERE — P34 submission, step by step

**Saumya Joshi · ADT23SOCB0997 · Roll 34 · AIA-05 Batch B · Deadline: Sat 7 Nov 2026**

## The map: what runs WHERE

| Place | What you do there | What comes out |
|---|---|---|
| **Google Colab** (browser) | Run 6 cells in `P34_colab_V2.ipynb`. It *scores* the AI on 20 questions. | Correctness %, PDF report |
| **huggingface.co** (browser) | Make a Space and drag-and-drop the project files in. | Your **live app link** |
| **github.com** (browser) | Make a repo and drag-and-drop the same files in. | Your **GitHub link** |
| **Your computer** | Nothing to install. You only download/upload files and record a video. | Demo video |
| **Google Drive** | Put the final files in folder `34_Saumya_Joshi`. | The submission |

Colab does **not** run the web app. The web app only runs on the Hugging Face Space.
You never need to run anything in CMD or install Python for this plan.

## The 6 steps (do them in this order)

### STEP 1 — Colab: get your real correctness % and PDF  (~20 min)
1. Open colab.research.google.com and sign in with Google.
2. **File → Upload notebook** → choose `P34_colab_V2.ipynb`.
3. Click ▶ on the cells labelled **CELL 1 … CELL 6**, one at a time, in order. Wait for a green tick each time.
   - CELL 2: a **Choose Files** button appears. Pick `p34_vqa_project_V2.zip` (leave it zipped).
4. STOP at the red "STOP HERE FOR NOW" heading.

You should now have downloaded 3 files: `P34_report_Saumya_Joshi.pdf`, `results.json`, `results_summary.md`.
Keep the Colab tab open — you will come back to it in Step 4.

### STEP 2 — Hugging Face: make the live app  (~15 min, mostly waiting)
1. huggingface.co → sign up / log in → verify e-mail.
2. Profile picture → **New Space** → Name `p34-vqa-app` · SDK **Gradio** · Template **Blank** · Hardware **CPU basic (free)** · **Public** → **Create Space**.
3. Unzip `p34_vqa_project_V2.zip` on your computer. Open the `p34_vqa` folder inside.
4. On the Space: **Files** tab → **Add file → Upload files**. Select **everything inside the `p34_vqa` folder** (all files and the folders `testset`, `docs`, `tests`) and drag it in.
   `README.md` must be at the TOP level of the Space, not inside a sub-folder.
5. **Commit changes to main**.
6. Wait. Open the **Logs** tab. After ~10–15 min the page says **Running**.
7. Your live link is the page address: `https://huggingface.co/spaces/YOUR_HF_USERNAME/p34-vqa-app`

### STEP 3 — GitHub: make the repo  (~5 min)
1. github.com/new → name `p34-vqa-app` → **Public** → tick NOTHING → **Create repository**.
2. Click **uploading an existing file** → drag in the same files and folders as Step 2 → **Commit changes**.
3. Your GitHub link: `https://github.com/YOUR_GITHUB_USERNAME/p34-vqa-app`

### STEP 4 — Colab: put the real links into the PDF  (~3 min)
1. Go back to the Colab tab. If it has disconnected, re-run CELL 1, CELL 2, CELL 4 first (it recreates the results).
2. Scroll to **LATER A**. Replace the two links with your real ones. Click ▶.
3. Click ▶ on **LATER B** to download the final PDF.
4. On GitHub: **Add file → Upload files**. First type `results/` at the start of the file name box, then upload `P34_report_Saumya_Joshi.pdf`, `results.json`, `results_summary.md`.
5. On GitHub and on the Space, open `README.md` (pencil icon) and replace `HF_USER` in the **Live app** line with your real username. Commit.

### STEP 5 — Record the 2–3 minute demo video  (~15 min)
Use `docs/DEMO_VIDEO_SCRIPT.md`. Show your live Space link in the address bar. Say the real numbers from the PDF.

### STEP 6 — Submit in Google Drive  (~5 min)
Make a folder named exactly `34_Saumya_Joshi` and put in it:
1. `P34_report_Saumya_Joshi.pdf`  (the final one from LATER B)
2. `LINKS.txt`  (fill in both links)
3. your demo video (`.mp4`, 2:00–3:00)

Then tick everything in `docs/SUBMISSION_CHECKLIST.md`.

## If something goes wrong
| Problem | Fix |
|---|---|
| Red text in Colab CELL 1 | Normal. Only `Traceback` or `FAIL` matters. |
| Colab disconnected | Re-run CELL 1, CELL 2 and CELL 4. |
| Space says **Build error** | Logs tab → copy the last 30 lines → send them to me. |
| Space says **Configuration error** | `README.md` is not at the top level of the Space. |
| Space says **Sleeping** | Click **Restart**. Do this before your demo. |
| Accuracy looks low | Fine. The task wants a measured result and a failure study, not a high score. Never edit numbers. |
