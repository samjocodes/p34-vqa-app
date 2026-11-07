# Measured results (generated 2026-11-07 14:50)

- VQA model: `Salesforce/blip-vqa-base` on cpu
- LLM backend: Qwen/Qwen2.5-0.5B-Instruct (local)
- BERT-family scorer: `sentence-transformers/all-MiniLM-L6-v2`

| Condition | Pairs | Correct | Correct % |
|---|---|---|---|
| clean | 8 | 5 | 62.5 |
| lowlight | 6 | 5 | 83.3 |
| cluttered | 6 | 4 | 66.7 |
| **All** | 20 | 14 | **70.0** |

Low-light before/after brightening: 83.3% -> 83.3% (overall 70.0% -> 70.0%).

| Prompt | Faithful % | Fallback % | Mean similarity | Mean words |
|---|---|---|---|---|
| zero_shot | 85.0 | 15.0 | 0.843 | 6.9 |
| few_shot | 100.0 | 0.0 | 0.88 | 6.4 |

Bias test: gendered answers on gender-neutral silhouettes: 10/10 (100.0%).
Guardrails: 10/10 sensitive questions blocked, 10/10 benign questions allowed.

| ID | Condition | Question | Expected | Answer | Correct |
|---|---|---|---|---|---|
| c01 | clean | What color is the circle? | red | red | yes |
| c02 | clean | How many shapes are there? | 3 | 5 | NO |
| c03 | clean | How many squares are there? | 3 | 4 | NO |
| c04 | clean | Is there a triangle in the image? | yes | yes | yes |
| c05 | clean | Is there a triangle in the image? | no | no | yes |
| c06 | clean | What shape is the blue object? | triangle | triangle | yes |
| c07 | clean | What color is the circle? | yellow | yellow | yes |
| c08 | clean | How many circles are there? | 2 | 3 | NO |
| l01 | lowlight | What color is the circle? | red | red | yes |
| l02 | lowlight | How many squares are there? | 2 | 3 | NO |
| l03 | lowlight | Is there a circle in the image? | yes | yes | yes |
| l04 | lowlight | What shape is the green object? | triangle | triangle | yes |
| l05 | lowlight | Is there a triangle in the image? | no | no | yes |
| l06 | lowlight | What color is the triangle? | red | red | yes |
| k01 | cluttered | What color is the circle? | blue | blue | yes |
| k02 | cluttered | How many triangles are there? | 2 | 5 | NO |
| k03 | cluttered | Is there a square in the image? | yes | yes | yes |
| k04 | cluttered | What shape is the yellow object? | square | square | yes |
| k05 | cluttered | Is there a triangle in the image? | no | no | yes |
| k06 | cluttered | How many circles are there? | 3 | 4 | NO |
