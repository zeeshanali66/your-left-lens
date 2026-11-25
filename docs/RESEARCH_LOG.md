# Research log: does the model know YOUR left?

### 1. Why
Three earlier studies in this series (whose-left, says-vs-does, your-left-or-mine) kept finding the same error: models
hand over "my left" as "your left". This study asks where that happens inside the model, and starts with a cleaner
behaviour test than before: one sentence of relation (same way / facing each other), one question, one word answer.

### 2. Design decisions
- Four conditions, so that a fixed rule can be told apart from real perspective-taking. `same` (beside me, same way:
  your left = my left) is the key one: a model that simply flips for "the other person" fails it, and earlier studies
  never tested it.
- `self_same` and `self_opp` ask about the speaker's own side; the answer is always the stated side. They show whether
  the model reads the sentence at all.
- Four templates (table, two named people, robot and user, two friends) so that probes and the early-exit layer can be
  chosen on some templates and tested on others.
- Novelty check (2026): closest work is arXiv 2606.01914 (spatial answer words; the correct relation is available
  inside the model; patching), perspective benchmarks 2603.23510 and 2601.16378 (behaviour or fine-tuning), and a
  103-model VLM study that reports egocentric bias. None looks inside text LLMs at the left/right perspective flip.

### 3. Problems and fixes
- Patching first used `hidden_states` as the source, whose last entry may already be normalised. Switched to hooks on
  the same layer modules, and added a hard check: patching the last layer must reproduce the source's own final logit
  difference (it matches exactly: -2.5 = -2.5 for Gemma 4 E4B, 4.469 = 4.469 for Qwen).
- The first patching measure ("did the answer switch") was uninformative when both runs give the same answer; it was
  replaced by a normalised effect that skips pairs whose outputs differ by less than 1 logit.
- Gemma 4 E4B runs in 4-bit with its per-layer embedding table on the CPU (it does not fit otherwise). That made
  patching (86 forward passes per pair) far too slow: the patching and prompt-fix steps were stopped for this model.
  Patching was kept for Qwen2.5-1.5B, but only 1 pair passed the 1-logit filter, because the relation sentence barely
  changes Qwen's output either. So **patching is not informative in this study**, and this is itself evidence for the
  main finding.
- The first Ollama behaviour run failed when Ollama's model server crashed on load (CUDA init error, probably low
  system RAM while another job ran). The client now retries and then stops instead of recording an empty answer.
- A duplicate queue process wrote to the Qwen output folder at the same time as the real run; the folder was deleted
  and the Qwen run repeated cleanly.

### 4. Reading the result
The behaviour result is the headline: models use a fixed "mirror for the other person" rule. The inside view agrees:
the facing sentence hardly moves the output, and the other person's correct side is not linearly decodable at any
layer, while the speaker's own side is. The Gemma 4 mid-layer signal for `same` items (0.62 at layer 28, then 0.28 at
the output) is real but weak, and an early-exit layer chosen on held-out templates does not fix anything, so it is
reported as a hint, not as a "the model knows but overwrites it" result.

## Next
Patching on a model that runs fully on the GPU; a prompt that states the rule as a worked example; whether
fine-tuning on a few `same`-way items removes the fixed rule or only moves it.
