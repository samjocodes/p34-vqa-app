# Demo video script (target 2:30, hard limits 2:00–3:00)

Run the **Evaluation tab once BEFORE recording** so the results and the PDF are ready. Open the live Space URL first so models are already loaded (otherwise the first question takes minutes). Keep the browser tab on the live URL visible (address bar showing the URL).

| Time | Show | Say (adapt in your own words) |
|---|---|---|
| 0:00–0:15 | Title / live app page | "Hi, I'm Saumya Joshi, roll number 34, AIA-05 Batch B. This is my project P34, a Visual Question Answering app, deployed live on Hugging Face Spaces." |
| 0:15–1:00 | Ask tab: upload a generated image (shapes `square:red, triangle:blue, circle:green`, clean), template *Identity* (`blue`) → Ask | "U1: question templates. U3: BLIP-VQA answers 'triangle'. U2: an LLM expands it into a full sentence. The probability view shows how confident the model is." |
| 1:00–1:30 | Generator tab: lowlight image → send to Ask → ask → show warning/auto-brighten | "U4: I generate hard test images with known ground truth. This one is low-light; the app brightens it automatically and warns when confidence is low." |
| 1:30–1:50 | Ask: "Is this person a man or a woman?" → refused | "Responsible AI: the app refuses sensitive-attribute questions. My bias test showed the model guesses gender even for neutral silhouettes." |
| 1:50–2:30 | Evaluation tab results + open the PDF (failure case page) | "U5: correctness on the 20 image–question pairs is X%. Low-light was Y% before and Z% after brightening. This is a failure case. Accessibility: this could help low-vision users but dark or cluttered photos are where it is least reliable." |
| 2:30–2:45 | Layers recap (README table) | "The project covers all five layers: lifecycle and prompt comparison, transformer LLM, GPT-style and BERT-family models with images, image generation, and responsible AI with a model card. Thank you." |

Replace X, Y, Z with the real numbers from your own Evaluation tab. Don't read the script word for word – speak naturally.
