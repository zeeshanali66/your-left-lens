# Does the model know YOUR left? A fixed "mirror" rule for the other person, inside and out

Earlier studies in this series kept finding one error: models give "my left" as "your left" when two people face
each other. This study tests the reverse case that earlier work skipped, and then looks inside two models.

> "You sit **next to me, facing the same way**. The cup is on my left. Is it on **your** left or right?"
> Correct: left. Almost every model says **right**.

## Design
- **320 items**: 4 templates (table, two named people, robot and user, two friends) x 10 objects x 2 sides x 4
  conditions. One-word answer (left/right); chance is 0.50.
  - `same`: beside me, facing the same way; asks about *your* side. Answer = the stated side.
  - `opposite`: across from me, facing me; asks about *your* side. Answer = the flipped side.
  - `self_same`, `self_opp`: the same two scenes, but asks about *my* side. Answer = the stated side.
- **Behaviour**: Gemma 4 12B, Qwen 3.5 9B, Gemma 3 12B, Qwen 2.5 7B, Llama 3.1 8B (Ollama, temperature 0);
  1,600 answers, all parsed.
- **Inside the model** (Gemma 4 E4B and Qwen2.5-1.5B, Hugging Face): logit lens at every layer; linear probes for the
  correct side (trained on some templates, tested on others); activation patching between matched same/facing items;
  three fixes (early exit at a layer chosen on held-out templates, a step-by-step prompt, a rule hint).

## Results

### 1. Every model uses a fixed rule: "the other person's side is the opposite of mine"

| model | same (answer: unflipped) | opposite (answer: flipped) | self_same | self_opp | says "flipped" for the other person |
|---|---|---|---|---|---|
| Gemma 4 12B | **0.01** [0.00, 0.04] | 0.97 | 0.99 | 0.75 | 0.98 |
| Qwen 3.5 9B | 0.38 [0.28, 0.49] | 0.96 | 1.00 | 0.94 | 0.79 |
| Gemma 3 12B | 0.30 [0.20, 0.40] | 0.89 | 0.79 | 0.72 | 0.79 |
| Qwen 2.5 7B | 0.09 [0.04, 0.15] | 0.97 | 1.00 | 0.99 | 0.94 |
| Llama 3.1 8B | 0.29 [0.19, 0.39] | 0.74 | 0.72 | 0.66 | 0.72 |

Every model is **below chance** on `same` (all 95% CIs exclude 0.50) and near ceiling on `opposite`. Models are not
taking the other person's perspective; they flip left and right whenever the question is about the other person,
whether or not that person faces them. The strongest model, Gemma 4, is the most rigid (0.98 flipped). Earlier studies
that tested only the facing case would have counted this rule as success.

### 2. Inside the model, the facing sentence barely matters
In Gemma 4 E4B, the logit difference for the unflipped side is -0.96 when facing the same way and -1.32 when facing
each other: the relation sentence moves the output by only 0.36 logits, and in half of the matched pairs it moves it
the wrong way. Qwen2.5-1.5B is at chance on both conditions and the relation moves it by -0.32.

### 3. The other person's side is not represented; one's own side is
Linear probes, trained on some templates and tested on others, decode the correct answer for the speaker's own side
almost perfectly in Gemma 4 E4B (up to 0.97 at layer 24), but never decode the correct answer for the other person
above 0.64 in any layer (`same`) or 0.57 (`opposite`). The model does not seem to build a representation of "your
left" at all; it applies a word-level rule.

### 4. A weak mid-layer hint, not a hidden answer
For `same` items, Gemma 4 E4B's logit lens favours the correct (unflipped) side at layer 28 (0.62) before the last
layers turn to the flip (0.28 at the output). But an early-exit layer chosen on held-out templates does not improve
anything, so this is a hint, not evidence that the model "knows and overwrites".

### 5. Patching was not informative here
Patching needs pairs where the relation changes the output; in Qwen only 1 pair passed the 1-logit filter, and the
step was too slow for Gemma 4 E4B (part of the model runs on the CPU), so it was stopped. That the relation rarely
changes the output is itself consistent with results 1 and 2.

### Fixes (Qwen2.5-1.5B, held-out templates)
Neither early exit, a step-by-step prompt (0.48 overall) nor a rule hint (0.50) helped the small model.

## Why it matters
Robots and agents that talk about space ("hand me the one on your left") need the listener's frame, not a fixed
mirror. Benchmarks that only test people facing each other reward the mirror rule. Testing the same-way case is cheap
and exposes it.

## Reproduce
```
python lens_study.py items               # builds data/items.jsonl
python lens_study.py ollama              # behaviour, five models
python lens_study.py run gemma4-e4b      # lens, probes (SKIP_GEN=1 to skip prompt fixes)
python lens_study.py run qwen1.5b        # lens, probes, patching, fixes
python lens_study.py results             # results/RESULTS.md
```
Hardware: RTX 3060 Ti (8 GB). `watchdog.py` stops a run if the GPU reaches 84 C.

## Limitations
Text only, one-sentence scenes; one answer format; two models inside; Gemma 4 E4B in 4-bit; no patching result; no
human baseline (people also make left/right errors, but not this systematic rule).

## Related work
arXiv 2606.01914 (spatial answer words, relation available internally, patching); 2603.23510 and 2601.16378
(perspective-taking benchmarks and fine-tuning); Halawi et al., "Overthinking the Truth" (correct answers in middle
layers). This study adds the same-way control, which shows the error is a fixed rule rather than a failed
perspective computation.

License: MIT.
