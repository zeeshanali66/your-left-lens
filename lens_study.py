"""Does the model know YOUR left? Looking inside LLMs at the perspective flip.

Earlier pilots in this series found the same error again and again: when two people face each other, models give
"my left" as "your left" (the director in your-left-or-mine-agents, the robot in says-vs-does, whose-left).
This study asks WHERE that error happens inside the model:
  (a) the model never computes the flip, or
  (b) it computes the flip in middle layers and later layers overwrite it with the copied word ("overthinking").
Items (one-word answer, left/right), 10 objects x 2 sides x 4 templates:
  same      - you sit next to me, facing the same way; "is it on YOUR left or right?"     answer = stated side
  opposite  - you sit across from me, facing me;       "is it on YOUR left or right?"     answer = flipped side
  self_opp  - you sit across from me, facing me;       "is it on MY left or right?"       answer = stated side
              (control: the relation is there but irrelevant; a model that flips whenever it sees "facing me"
               fails here)
Measures (logit difference correct minus wrong, first answer token):
  behaviour    - final-layer accuracy per condition
  logit lens   - accuracy / mean logit difference at every layer
  probe        - linear probe for the correct side on the last-token residual, leave-one-template-out
  patching     - copy the last-token residual at layer l from the opposite run into the matched same run:
                 from which layer on does the answer switch?
  fixes        - early exit at a layer chosen on templates 0-1 (tested on 2-3), step-by-step prompt, rule hint
Commands: items | pilot <model> | run <model> | results
"""
import json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, RES = os.path.join(HERE, "data"), os.path.join(HERE, "results")
MODELS = {"qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "gemma4-e4b": "google/gemma-4-E4B-it"}
OBJECTS = ["cup", "book", "phone", "lamp", "bottle", "apple", "pen", "plate", "key", "candle"]
CONDS = ["same", "opposite", "self_same", "self_opp"]
FLIP = {"left": "right", "right": "left"}
# each template: (relation sentence for same, for opposite, statement, question about you, question about me)
TEMPLATES = [
    ("You are sitting next to me at the table, facing the same direction as me.",
     "You are sitting across the table from me, facing me.",
     "From where I sit, the {o} is on my {s}.",
     "From where you sit, is the {o} on your left or your right?",
     "From where I sit, is the {o} on my left or my right?"),
    ("Ali and Sara stand side by side, both facing the same wall.",
     "Ali and Sara stand face to face, looking at each other.",
     "There is a {o} on Ali's {s}.",
     "Is the {o} on Sara's left or on Sara's right?",
     "Is the {o} on Ali's left or on Ali's right?"),
    ("I am a robot. The user is standing beside me and looking the same way I am looking.",
     "I am a robot. The user is standing in front of me and looking at me.",
     "My camera sees the {o} on my {s} side.",
     "For the user, is the {o} on their left or their right?",
     "For me, the robot, is the {o} on my left or my right?"),
    ("Picture two friends on a bench, both facing the sea.",
     "Picture two friends sitting opposite each other at a small cafe table.",
     "The first friend says: \"The {o} is on my {s}.\"",
     "Is the {o} on the second friend's left or right?",
     "Is the {o} on the first friend's left or right?"),
]
SUFFIX = "\nAnswer with one word: left or right."
HINT = (" (Rule: if two people face the same way, left and right are the same for both; if they face each other, "
        "one person's left is the other person's right.)")
COT = "\nThink step by step in one or two short sentences, then end with 'Answer: left' or 'Answer: right'."


def make_items():
    items = []
    for t, (rs, ro, st, qy, qm) in enumerate(TEMPLATES):
        for o in OBJECTS:
            for s in ("left", "right"):
                for c in CONDS:
                    rel = rs if c in ("same", "self_same") else ro
                    q = qm if c.startswith("self") else qy
                    ans = FLIP[s] if c == "opposite" else s
                    items.append({"id": f"t{t}-{o}-{s}-{c}", "tpl": t, "obj": o, "side": s, "cond": c, "answer": ans,
                                  "text": f"{rel} {st.format(o=o, s=s)} {q.format(o=o)}"})
    return items


def cmd_items():
    items = make_items()
    assert len(items) == 4 * 10 * 2 * 4 and len({i["id"] for i in items}) == len(items)
    for i in items:   # sanity: exactly one relation sentence, and the stated side, appear in the text
        t = TEMPLATES[i["tpl"]]
        assert (t[0] in i["text"]) != (t[1] in i["text"])
        assert (t[1] in i["text"]) == (i["cond"] in ("opposite", "self_opp"))
        assert f"{i['obj']} is on" in i["text"] or i["obj"] in i["text"]
    # answers are balanced: half left, half right in every condition
    for c in CONDS:
        a = [i["answer"] for i in items if i["cond"] == c]
        assert a.count("left") == a.count("right")
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "items.jsonl"), "w", encoding="utf-8") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")
    print(f"items ok: {len(items)} ({len(TEMPLATES)} templates x {len(OBJECTS)} objects x 2 sides x 4 conditions);",
          "answers balanced per condition")
    for c in CONDS:
        print(" ", c, "|", next(i["text"] for i in items if i["cond"] == c and i["tpl"] == 0))


# ---------------------------------------------------------------- model helpers
def load(key, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    if key == "gemma4-e4b" and device == "cuda":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
                               llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            name, quantization_config=q, dtype=torch.bfloat16,
            device_map={"model.language_model.embed_tokens_per_layer": "cpu", "": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        model.to(device)
    model.eval()
    return tok, model


def parts(model):
    """(decoder layers ModuleList, final norm, lm_head) for Qwen2 and Gemma 4."""
    import torch.nn as nn
    best = None
    for name, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and name.endswith("layers") and "vision" not in name and "audio" not in name:
            if best is None or len(m) > len(best[1]):
                best = (name, m)
    base = best[0].rsplit(".", 1)[0]
    norm = model.get_submodule(base + ".norm")
    return best[1], norm, model.get_output_embeddings()


def answer_ids(tok):
    ids = {}
    for side in ("left", "right"):
        cand = set()
        for v in (side, side.capitalize(), " " + side, " " + side.capitalize()):
            t = tok.encode(v, add_special_tokens=False)
            if t:
                cand.add(t[0])
        ids[side] = sorted(cand)
    assert not set(ids["left"]) & set(ids["right"])
    return ids


def prompt(tok, text, extra=SUFFIX):
    return tok.apply_chat_template([{"role": "user", "content": text + extra}], add_generation_prompt=True,
                                   tokenize=False)


def ld_from_logits(logits, ids, ans):
    """logit difference (correct minus wrong), max over spelling variants."""
    c = logits[ids[ans]].max().item()
    w = logits[ids[FLIP[ans]]].max().item()
    return c - w


def forward(tok, model, text, extra=SUFFIX):
    import torch
    enc = tok(prompt(tok, text, extra), return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    return enc, out


def lens_row(tok, model, item, ids, layers, norm, head):
    """final LD, LD at every layer (logit lens), last-token residuals."""
    import torch
    enc, out = forward(tok, model, item["text"])
    hs = out.hidden_states            # embeddings + one per layer; the last one is already normed
    lds, resid = [], []
    with torch.no_grad():
        for l in range(1, len(hs)):
            h = hs[l][0, -1]
            resid.append(h.float().cpu())
            hn = h if l == len(hs) - 1 else norm(h.unsqueeze(0))[0]
            lg = head(hn.unsqueeze(0).to(head.weight.dtype))[0].float()
            lds.append(ld_from_logits(lg, ids, item["answer"]))
    final = ld_from_logits(out.logits[0, -1].float(), ids, item["answer"])
    return final, lds, torch.stack(resid)


def patch_curve(tok, model, src_item, dst_item, ids, layers):
    """Put src's last-token output of layer l into dst's run; return dst LD w.r.t. SRC's answer, per l.
    Source states are recorded with hooks on the same layer modules, so patching the last layer reproduces the
    source's final logits exactly."""
    import torch
    src_h = [None] * len(layers)

    def rec(l):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            src_h[l] = h[0, -1].detach().clone()
        return hook
    hds = [layer.register_forward_hook(rec(l)) for l, layer in enumerate(layers)]
    try:
        enc_s = tok(prompt(tok, src_item["text"]), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            model(**enc_s)
    finally:
        for h in hds:
            h.remove()
    enc = tok(prompt(tok, dst_item["text"]), return_tensors="pt", add_special_tokens=False).to(model.device)
    res = []
    for l, layer in enumerate(layers):
        def hook(mod, inp, out, l=l):
            h = out[0] if isinstance(out, tuple) else out
            h[0, -1] = src_h[l].to(h.dtype)
            return out
        hd = layer.register_forward_hook(hook)
        try:
            with torch.no_grad():
                lg = model(**enc).logits[0, -1].float()
        finally:
            hd.remove()
        res.append(ld_from_logits(lg, ids, src_item["answer"]))
    return res


def gen_answer(tok, model, text, extra, max_new=80):
    import torch
    enc = tok(prompt(tok, text, extra), return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    s = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    m = re.findall(r"\b(left|right)\b", s.lower())
    return (m[-1] if m else None), s


# ---------------------------------------------------------------- commands
def items_all():
    p = os.path.join(DATA, "items.jsonl")
    if not os.path.exists(p):
        cmd_items()
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    if os.environ.get("SMOKE"):   # tiny subset to test the code paths
        rows = [r for r in rows if r["obj"] == "cup"]
    return rows


def cmd_pilot(key):
    """CPU feasibility check on 24 items: behaviour + logit lens only."""
    tok, model = load(key, "cpu")
    layers, norm, head = parts(model)
    ids = answer_ids(tok)
    items = [i for i in items_all() if i["obj"] in ("cup", "lamp")]
    t0 = time.time()
    for c in CONDS:
        rows = [lens_row(tok, model, i, ids, layers, norm, head) for i in items if i["cond"] == c]
        fin = [r[0] > 0 for r in rows]
        per_layer = [sum(r[1][l] > 0 for r in rows) / len(rows) for l in range(len(layers))]
        print(f"{c:9s} final acc {sum(fin) / len(fin):.2f}  lens acc by layer:",
              " ".join(f"{a:.1f}" for a in per_layer), flush=True)
    print(f"{len(items)} items, {time.time() - t0:.0f}s, {len(layers)} layers, answer ids {ids}")


def cmd_run(key):
    import numpy as np, torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok, model = load(key, device)
    layers, norm, head = parts(model)
    ids = answer_ids(tok)
    items = items_all()
    out_dir = os.path.join(RES, key)
    os.makedirs(out_dir, exist_ok=True)
    # 1. behaviour + logit lens + residuals
    lens_p = os.path.join(out_dir, "lens.jsonl")
    if not os.path.exists(lens_p):
        resid = []
        with open(lens_p, "w", encoding="utf-8") as f:
            for n, it in enumerate(items):
                final, lds, r = lens_row(tok, model, it, ids, layers, norm, head)
                resid.append(r.numpy().astype(np.float16))
                f.write(json.dumps({"id": it["id"], "final": final, "lens": lds}) + "\n")
                if n % 60 == 0:
                    print(key, "lens", n, flush=True)
        np.save(os.path.join(out_dir, "resid.npy"), np.stack(resid))
    # 2. patching: opposite -> same (does the answer flip?) and same -> opposite, matched pairs
    patch_p = os.path.join(out_dir, "patch.jsonl")
    if not os.path.exists(patch_p):
        by = {i["id"]: i for i in items}
        a, b = items[0], items[1]
        src_final = lens_row(tok, model, a, ids, layers, norm, head)[0]
        got = patch_curve(tok, model, a, b, ids, layers)[-1]
        assert abs(got - src_final) < 0.05 * max(1.0, abs(src_final)), (got, src_final)
        print(key, "patch check ok", round(got, 3), round(src_final, 3), flush=True)
        with open(patch_p, "w", encoding="utf-8") as f:
            for n, it in enumerate([i for i in items if i["cond"] == "opposite"]):
                twin = by[it["id"].replace("-opposite", "-same")]
                f.write(json.dumps({"id": it["id"], "opp_into_same": patch_curve(tok, model, it, twin, ids, layers),
                                    "same_into_opp": patch_curve(tok, model, twin, it, ids, layers)}) + "\n")
                if n % 20 == 0:
                    print(key, "patch", n, flush=True)
    # 3. prompt fixes (generation); SKIP_GEN=1 skips it (Gemma 4 with CPU-offloaded embeddings is too slow)
    if os.environ.get("SKIP_GEN"):
        print(key, "done (generation skipped)", flush=True)
        return
    gen_p = os.path.join(out_dir, "gen.jsonl")
    done = {(r["id"], r["mode"]) for r in map(json.loads, open(gen_p, encoding="utf-8"))} if os.path.exists(gen_p) else set()
    with open(gen_p, "a", encoding="utf-8") as f:
        for n, it in enumerate(items):
            for mode, extra, mx in (("cot", COT, 90), ("hint", HINT + SUFFIX, 6)):
                if (it["id"], mode) in done:
                    continue
                a, s = gen_answer(tok, model, it["text"], extra, mx)
                f.write(json.dumps({"id": it["id"], "mode": mode, "pred": a, "ok": a == it["answer"],
                                    "raw": s[:300]}) + "\n")
            if n % 60 == 0:
                print(key, "gen", n, flush=True)
    print(key, "done", flush=True)


OLLAMA_MODELS = ["gemma4:12b", "qwen3.5:9b", "gemma3:12b", "qwen2.5:7b", "llama3.1:8b"]


def cmd_ollama():
    """Behaviour only, on the five chat models used in the rest of the series (one-word answer)."""
    import urllib.request
    p = os.path.join(RES, "ollama.jsonl")
    done = {(r["model"], r["id"]) for r in map(json.loads, open(p, encoding="utf-8"))} if os.path.exists(p) else set()
    os.makedirs(RES, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for m in OLLAMA_MODELS:
            for it in items_all():
                if (m, it["id"]) in done:
                    continue
                extra = {"think": False} if m.startswith(("qwen3", "gemma4")) else {}
                body = json.dumps({"model": m, "stream": False, **extra, "messages": [
                    {"role": "user", "content": it["text"] + SUFFIX}],
                    "options": {"temperature": 0, "seed": 42, "num_predict": 8}}).encode()
                req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                             headers={"Content-Type": "application/json"})
                for attempt in range(3):
                    try:
                        raw = json.loads(urllib.request.urlopen(req, timeout=900).read())["message"]["content"]
                        break
                    except Exception as e:
                        print("retry", m, it["id"], e, flush=True); time.sleep(60)
                else:
                    raise RuntimeError("Ollama failed 3 times")
                mm = re.findall(r"\b(left|right)\b", raw.lower())
                a = mm[0] if mm else None
                f.write(json.dumps({"model": m, "id": it["id"], "pred": a, "ok": a == it["answer"], "raw": raw[:60]}) + "\n")
                f.flush()
            print("ollama", m, "done", flush=True)


def cmd_results():
    import numpy as np, pandas as pd
    from sklearn.linear_model import LogisticRegression
    items = pd.DataFrame(items_all())
    out = ["# Does the model know YOUR left? Results", ""]
    rng = np.random.default_rng(0)

    def boot(x):
        x = np.asarray(x, float)
        b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
        return f"{x.mean():.2f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]"

    op = os.path.join(RES, "ollama.jsonl")
    if os.path.exists(op):
        O = items.merge(pd.DataFrame([json.loads(l) for l in open(op, encoding="utf-8")]), on="id")
        out += ["## Behaviour of five chat models (Ollama, one-word answer; chance 0.50)", "",
                "| model | " + " | ".join(CONDS) + " | answers 'flipped' when asked about the OTHER person |",
                "|---|" + "---|" * (len(CONDS) + 1)]
        for m in OLLAMA_MODELS:
            g = O[O.model == m]
            if not len(g):
                continue
            oth = g[g.cond.isin(["same", "opposite"])]
            flip = (oth.pred == oth.side.map(FLIP)).mean()
            out.append(f"| {m} | " + " | ".join(boot(g[g.cond == c].ok) for c in CONDS) + f" | {flip:.2f} |")
        out.append("")
    for key in MODELS:
        d = os.path.join(RES, key)
        if not os.path.exists(os.path.join(d, "lens.jsonl")):
            continue
        lens = pd.DataFrame([json.loads(l) for l in open(os.path.join(d, "lens.jsonl"), encoding="utf-8")])
        df = items.merge(lens, on="id")
        L = len(df.lens.iloc[0])
        out += [f"## {key} ({L} layers)", "", "### Behaviour (final layer accuracy, 95% bootstrap CI; chance 0.50)", "",
                "| condition | accuracy | mean logit diff (correct - wrong) |", "|---|---|---|"]
        for c in CONDS:
            g = df[df.cond == c]
            out.append(f"| {c} | {boot(g.final > 0)} | {g.final.mean():+.2f} |")
        # logit lens
        out += ["", "### Logit lens: accuracy at each layer", "", "```",
                "layer    " + " ".join(f"{l:>4d}" for l in range(L))]
        for c in CONDS:
            g = df[df.cond == c]
            acc = [np.mean([x[l] > 0 for x in g.lens]) for l in range(L)]
            out.append(f"{c:9s}" + " ".join(f"{a:4.2f}" for a in acc))
        out.append("```")
        for c in ("same", "opposite"):
            g = df[df.cond == c]
            peak = max(range(L), key=lambda l: np.mean([x[l] > 0 for x in g.lens]))
            out.append(f"\n{c} items: best layer {peak} (accuracy {np.mean([x[peak] > 0 for x in g.lens]):.2f}) vs "
                       f"final {np.mean(g.final > 0):.2f}. Right at layer {peak} but wrong at the output: "
                       f"{np.mean([(x[peak] > 0) and (f <= 0) for x, f in zip(g.lens, g.final)]):.2f}.")
        # does the relation sentence change the output at all? matched pairs; logit for the UNflipped side
        by = df.set_index("id")
        sids = list(df[df.cond == "same"].id)
        sh = np.array([by.loc[i, "final"] for i in sids])
        op = np.array([-by.loc[i.replace("-same", "-opposite"), "final"] for i in sids])
        out.append(f"\nEffect of the relation sentence on the output (matched pairs): logit difference for the UNflipped "
                   f"side is {sh.mean():+.2f} when facing the same way and {op.mean():+.2f} when facing each other "
                   f"(difference {sh.mean() - op.mean():+.2f}; 0 would mean the relation is ignored; share of pairs "
                   f"where facing the same way favours the unflipped side more: {np.mean(sh > op):.2f}).")
        # probe
        R = np.load(os.path.join(d, "resid.npy")).astype(np.float32)
        y = (df.answer == "left").values.astype(int)
        tpl = df.tpl.values
        out += ["", "### Linear probe for the correct side (leave-one-template-out accuracy)", "", "```",
                "layer    " + " ".join(f"{l:>4d}" for l in range(0, L, 2))]
        accs = {c: [] for c in CONDS}
        for l in range(0, L, 2):
            pred = np.zeros(len(y))
            for t in range(len(TEMPLATES)):
                tr, te = tpl != t, tpl == t
                X = R[:, l]
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
                clf = LogisticRegression(max_iter=2000, C=0.1).fit((X[tr] - mu) / sd, y[tr])
                pred[te] = clf.predict((X[te] - mu) / sd)
            for c in CONDS:
                m = (df.cond == c).values
                accs[c].append((pred[m] == y[m]).mean())
        for c in CONDS:
            out.append(f"{c:9s}" + " ".join(f"{a:4.2f}" for a in accs[c]))
        out.append("```")
        # patching
        pp = os.path.join(d, "patch.jsonl")
        if os.path.exists(pp):
            P = [json.loads(l) for l in open(pp, encoding="utf-8")]
            fin = dict(zip(df.id, df.final))
            # normalised patching effect: 0 = dst unchanged, 1 = dst now behaves exactly like src
            # (LD measured w.r.t. the source's answer; source and destination answers are always opposite)
            def eff(r, direction):
                o_id, s_id = r["id"], r["id"].replace("-opposite", "-same")
                src, dst = (o_id, s_id) if direction == "opp_into_same" else (s_id, o_id)
                lo, hi = -fin[dst], fin[src]
                if hi - lo < 1.0:
                    return None          # the two runs barely differ: nothing to restore
                return [(v - lo) / (hi - lo) for v in r[direction]]
            rowsP = {}
            for direction in ("opp_into_same", "same_into_opp"):
                e = [x for x in (eff(r, direction) for r in P) if x is not None]
                rowsP[direction] = (np.mean(e, 0) if e else None, len(e))
            out += ["", "### Patching the last-token output of each layer (normalised effect: 0 = no change, "
                    "1 = the run now behaves like the source)", "", "```",
                    "layer         " + " ".join(f"{l:>5d}" for l in range(L))]
            for direction, lab in (("opp_into_same", "opp -> same "), ("same_into_opp", "same -> opp ")):
                m, n = rowsP[direction]
                out.append(lab + " " + (" ".join(f"{v:5.2f}" for v in m) if m is not None else "n/a") + f"   (n={n})")
            out += ["```", "Pairs where the same-way and facing runs differ by less than 1 logit are skipped: if the "
                    "relation does not change the output, there is nothing for patching to move. Reading: in early layers "
                    "the last token has not yet read the relation, so patching it changes little (near 0); the curve "
                    "rises to 1 at the layers where the relation's effect has arrived at the answer position."]
        # fixes
        dev, test = df.tpl.isin([0, 1]), df.tpl.isin([2, 3])
        best = max(range(L), key=lambda l: np.mean([x[l] > 0 for x in df[dev].lens]))
        out += ["", "### Fixes (templates 2-3 only; early-exit layer chosen on templates 0-1)", "",
                "| method | " + " | ".join(CONDS) + " | all |", "|---|" + "---|" * (len(CONDS) + 1)]

        def row(name, ok):
            t = df[test].assign(ok=ok)
            out.append(f"| {name} | " + " | ".join(f"{t[t.cond == c].ok.mean():.2f}" for c in CONDS) +
                       f" | {t.ok.mean():.2f} |")
        row("final layer (no fix)", (df[test].final > 0).values)
        row(f"early exit at layer {best}", np.array([x[best] > 0 for x in df[test].lens]))
        gp = os.path.join(d, "gen.jsonl")
        if os.path.exists(gp):
            G = pd.DataFrame([json.loads(l) for l in open(gp, encoding="utf-8")])
            for mode in ("cot", "hint"):
                g = df[test].merge(G[G["mode"] == mode][["id", "ok"]], on="id", how="left")
                row({"cot": "step-by-step prompt", "hint": "rule hint in the prompt"}[mode], g.ok.fillna(False).values)
        out.append("")
    os.makedirs(RES, exist_ok=True)
    open(os.path.join(RES, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "items"
    {"items": lambda: cmd_items(), "ollama": lambda: cmd_ollama(), "pilot": lambda: cmd_pilot(sys.argv[2]), "run": lambda: cmd_run(sys.argv[2]),
     "results": lambda: cmd_results()}[cmd]()
