# Deploy guide

The full step-by-step plan is in **`START_HERE.md`** (project root). Steps 2 and 3 there are the deployment.

Quick reference: no git and no CMD are needed. Everything is browser drag-and-drop.

- **Hugging Face Space:** New Space → Gradio · Blank · CPU basic (free) · Public → Files → Add file → Upload files → drag in everything inside the `p34_vqa` folder. `README.md` must be at the top level.
- **GitHub repo:** github.com/new → Public, tick nothing → "uploading an existing file" → drag in the same files.

## Optional: use git from the terminal instead
```
cd <folder that contains app.py>
git init
git add .
git commit -m "P34 VQA app"
git branch -M main
git remote add origin https://github.com/samjocodes/p34-vqa-app.git
git push -u origin main
git remote add space https://huggingface.co/spaces/samjocodes/p34-vqa-app
git push --force space main
```
For the Space push: Username = samjocodes, Password = a Hugging Face **Write** token (Settings → Access Tokens).

## Troubleshooting
| Message | Fix |
|---|---|
| `Build error` mentioning `huggingface-hub` | `requirements.txt` pins gradio 5 and transformers 4 on purpose. Do not change it. |
| `Configuration error` | `README.md` with its `---` header must be at the top level of the Space. |
| `Authentication failed` (git only) | Use a Write token as the password. |
| Space `Sleeping` | Click Restart. |
