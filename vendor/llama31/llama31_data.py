"""Data for the 20-task Llama-3.1-8B-Instruct LoRA bank (class-rich text port; brief
reports/studies/llama31_8b_run_brief_2026-09-29.md, protocol reports/studies/llama31_8b_protocol_2026-09-30.md).

Nested task lists (one bank, T8 subset T14 subset T20), brief section 2:
  T8  = Banking77 CLINC150(in-scope) LEDGAR MASSIVE-en MNLI ANLI MRPC QQP
  T14 = T8 + TREC-fine FewRel-64 DBpedia14 20Newsgroups AGNews QNLI
  T20 = T14 + RTE CoLA SST-2 SNLI SICK SciTail

Prompts: one template per format family, never a dataset name (leak_check). The user turn is wrapped in the
Llama-3 chat format by hand (the official chat template injects today's date in a system block).

Answers are MULTI-TOKEN (llama_data.py asserted single tokens): the answer sequence of a label is
tokens(label) + <|eot_id|>, so the label set of the bank is prefix-free and p(y|x) = prod_i p(y_i | x, y_<i) is the
exact probability of the complete answer. Score = SUM of the token log-probabilities (pre-registered, protocol
section 4). Label strings: labels.json (versioned), built by build_labels() from the datasets' own label names;
identical strings are the same class; only the brief's clean identities share strings (NLI-3, NLI-2, paraphrase).

Splits (identical construction to llama_data.py): evaluation source (official validation, else test) permuted with
seed 0, first 20 % = val (selection), remaining 80 % = test (reporting), capped at VAL_CAP / TEST_CAP. The train
split gives a HELD-OUT reserve of max(50, 3 C_t) examples (never shown to the expert; sized now for the brief's
second-phase pool control) and the expert's training set.
"""
from __future__ import annotations

import json
import os
import re
from functools import cache
from pathlib import Path

import numpy as np

STORE = Path("/shared/ssd_30T/azpitarte_llama31")
HF_HOME = STORE / "hf"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # official repo @ 0e9e39f2, safetensors sha256 in STORE/sha.txt
PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
N_PREFIX = 5  # tokens of PREFIX: identical activations for every prompt (causal), excluded from READ
EOT = 128009  # <|eot_id|>: terminates every answer sequence
PAD = 128004  # <|finetune_right_pad_id|> (the Meta tokenizer has no pad token); always masked

TASKS8 = ["banking77", "clinc", "ledgar", "massive", "mnli", "anli", "mrpc", "qqp"]
TASKS14 = TASKS8 + ["trec", "fewrel", "dbpedia", "news20", "agnews", "qnli"]
TASKS20 = TASKS14 + ["rte", "cola", "sst2", "snli", "sick", "scitail"]
TASKSETS = {8: TASKS8, 14: TASKS14, 20: TASKS20, 2: ["banking77", "mnli"]}  # T = 2: smoke mini-bank (item 9)
VAL_CAP, TEST_CAP = int(os.environ.get("LLAMA31_VAL_CAP", 250)), int(os.environ.get("LLAMA31_TEST_CAP", 1000))  # env: smoke
MAX_FIELD_TOKENS = 256  # long fields (clauses, documents, articles, premises) are cut to this many tokens
LABELS_PATH = Path(__file__).resolve().parent / "llama31_labels.json"

# task -> (loader key, train split, eval split, family)
SPECS = {
    "banking77": ("mteb/banking77", None, "train", "test", "intent"),
    "clinc": ("clinc/clinc_oos", "plus", "train", "validation", "intent"),
    "ledgar": ("coastalcph/lex_glue", "ledgar", "train", "validation", "text"),
    "massive": ("mteb/amazon_massive_intent", "en", "train", "validation", "intent"),
    "mnli": ("nyu-mll/glue", "mnli", "train", "validation_matched", "nli"),
    "anli": ("facebook/anli", None, "train", "dev", "nli"),  # rounds R1+R2+R3 concatenated (protocol 3.3)
    "mrpc": ("nyu-mll/glue", "mrpc", "train", "validation", "pair"),
    "qqp": ("nyu-mll/glue", "qqp", "train", "validation", "pair"),
    "trec": ("SetFit/TREC-QC", None, "train", "test", "question"),
    "fewrel": ("fewrel", None, "train", "eval", "relation"),  # train_wiki re-split (seed 3): 80 % train / 20 % eval
    "dbpedia": ("fancyzhx/dbpedia_14", None, "train", "test", "text"),
    "news20": ("sklearn20ng", None, "train", "test", "text"),
    "agnews": ("fancyzhx/ag_news", None, "train", "test", "text"),
    "qnli": ("nyu-mll/glue", "qnli", "train", "validation", "nli"),
    "rte": ("nyu-mll/glue", "rte", "train", "validation", "nli"),
    "cola": ("nyu-mll/glue", "cola", "train", "validation", "text"),
    "sst2": ("nyu-mll/glue", "sst2", "train", "validation", "text"),
    "snli": ("stanfordnlp/snli", None, "train", "validation", "nli"),
    "sick": ("parquet", None, "train", "validation", "nli"),
    "scitail": ("allenai/scitail", "snli_format", "train", "validation", "nli"),
}
DATASET_NAMES = ["MNLI", "ANLI", "SNLI", "QNLI", "RTE", "SICK", "SciTail", "MRPC", "QQP", "CoLA", "SST", "AGNews",
                 "AG News", "DBpedia", "Banking", "CLINC", "LEDGAR", "LexGLUE", "MASSIVE", "TREC", "FewRel",
                 "Newsgroups", "20NG", "Wikidata", "GLUE", "SuperGLUE"]
TEMPLATES = {
    "nli": "Premise:\n{a}\n\nHypothesis:\n{b}\n\nAnswer:",
    "pair": "Sentence 1:\n{a}\n\nSentence 2:\n{b}\n\nAnswer:",
    "text": "Text:\n{a}\n\nAnswer:",
    "intent": "Request:\n{a}\n\nAnswer:",
    "question": "Question:\n{a}\n\nAnswer:",
    "relation": "Sentence:\n{a}\n\nHead entity: {b}\nTail entity: {c}\n\nAnswer:",
}

# ------------------------------------------------------------------------------------------------ label strings
NLI3 = ["entailment", "neutral", "contradiction"]  # MNLI ANLI SNLI SICK (brief section 4, identity 1)
NLI2 = ["entailed", "not entailed"]              # QNLI RTE SciTail (identity 2; kept distinct from NLI3)
PARA = ["different", "same"]                     # MRPC QQP: not paraphrase, paraphrase (identity 3)
TREC_COARSE = {"ABBR": "abbreviation", "DESC": "description", "ENTY": "entity", "HUM": "human", "LOC": "location",
               "NUM": "numeric"}
TREC_FINE = {"abb": "abbreviation", "exp": "expansion", "body": "body part", "cremat": "creative work",
             "dismed": "disease", "instru": "instrument", "lang": "language", "techmeth": "technique",
             "termeq": "term", "veh": "vehicle", "def": "definition", "desc": "description", "gr": "group",
             "ind": "individual", "mount": "mountain", "dist": "distance", "ord": "rank", "perc": "percent",
             "temp": "temperature", "volsize": "size"}
NEWS20 = {"alt.atheism": "atheism", "comp.graphics": "computer graphics", "comp.os.ms-windows.misc": "windows os",
          "comp.sys.ibm.pc.hardware": "pc hardware", "comp.sys.mac.hardware": "mac hardware",
          "comp.windows.x": "x window system", "misc.forsale": "for sale", "rec.autos": "autos",
          "rec.motorcycles": "motorcycles", "rec.sport.baseball": "baseball", "rec.sport.hockey": "hockey",
          "sci.crypt": "cryptography", "sci.electronics": "electronics", "sci.med": "medicine", "sci.space": "space",
          "soc.religion.christian": "christianity", "talk.politics.guns": "gun politics",
          "talk.politics.mideast": "mideast politics", "talk.politics.misc": "politics",
          "talk.religion.misc": "religion talk"}
FIXED = {
    "mnli": NLI3, "anli": NLI3, "snli": NLI3, "sick": NLI3,
    "qnli": NLI2, "rte": NLI2, "scitail": NLI2,
    "mrpc": PARA, "qqp": PARA,
    "cola": ["incorrect", "correct"], "sst2": ["negative", "positive"],
    "agnews": ["world", "sports", "business", "technology"],
    "dbpedia": ["company", "school", "artist", "athlete", "official", "vehicle", "building", "nature", "town",
                "animal", "plant", "album", "film", "book"],
}
IDENTITY_GROUPS = [set(["mnli", "anli", "snli", "sick"]), set(["qnli", "rte", "scitail"]), set(["mrpc", "qqp"])]
# identical intent names given by two datasets themselves, same meaning (not a hand merge): one class of U, reported
SAME_STRING = {"exchange rate": {"banking77", "clinc"}, "play music": {"clinc", "massive"}}
# accidental identical strings between different meanings, resolved by renaming (protocol section 4); filled after
# the first collision report of build_labels()
RENAME = {}


def _intent(name):
    return name.replace("_", " ").strip().lower()


def build_labels():
    """task -> answer strings in label-index order, from the datasets' own label names (see RENAME)."""
    _env()
    out = dict(FIXED)
    out["banking77"] = _names("banking77")
    out["clinc"] = _names("clinc")
    out["massive"] = _names("massive")
    out["ledgar"] = _names("ledgar")
    out["trec"] = _names("trec")
    out["fewrel"] = _names("fewrel")
    out["news20"] = _names("news20")
    for t, ren in RENAME.items():
        out[t] = [ren.get(w, w) for w in out[t]]
    return {t: out[t] for t in TASKS20}


@cache
def labels():
    return json.loads(LABELS_PATH.read_text())["answers"]


def answers(task):
    return labels()[task]


def family(task):
    return SPECS[task][4]


def n_classes(task):
    return len(answers(task))


def heldout_size(task):
    return max(50, 3 * n_classes(task))


def _env():
    os.environ.setdefault("HF_HOME", str(HF_HOME))


@cache
def tokenizer():
    _env()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    assert tok.encode(PREFIX, add_special_tokens=False) == [128000, 128006, 882, 128007, 271]
    assert tok.encode(SUFFIX, add_special_tokens=False) == [128009, 128006, 78191, 128007, 271]
    assert tok.convert_ids_to_tokens(EOT) == "<|eot_id|>" and tok.convert_ids_to_tokens(PAD) == "<|finetune_right_pad_id|>"
    return tok


@cache
def answer_seq(word):
    """Token ids of a complete answer: tokens(word) + <|eot_id|>."""
    ids = tokenizer().encode(word, add_special_tokens=False)
    assert ids and EOT not in ids, word
    return tuple(ids) + (EOT,)


def union(tasks):
    """The bank's answer vocabulary U (ordered, unique strings), the answer token sequences of U, and per task the
    U-indices of H_t."""
    words = []
    for t in tasks:
        for w in answers(t):
            if w not in words:
                words.append(w)
    maps = {t: np.asarray([words.index(w) for w in answers(t)]) for t in tasks}
    return words, [answer_seq(w) for w in words], maps


# ----------------------------------------------------------------------------------------------- raw loading
@cache
def _hf(path, name, split):
    _env()
    from datasets import concatenate_datasets, load_dataset

    if path == "facebook/anli":
        return concatenate_datasets([load_dataset(path, split=f"{split}_r{r}") for r in (1, 2, 3)])
    return load_dataset(path, name, split=split)


@cache
def _fewrel():
    """(rows, relation ids in sorted order): train_wiki.json (64 relations x 700), re-split by _raw."""
    tw = json.loads((STORE / "data/fewrel/train_wiki.json").read_text())
    rels = sorted(tw)
    rows = [(r, ex) for r in rels for ex in tw[r]]
    return rows, rels


@cache
def _news20(split):
    from sklearn.datasets import fetch_20newsgroups

    b = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"), data_home=str(STORE / "data/sklearn"),
                           download_if_missing=False)
    return b.data, b.target, b.target_names


def _names(task):
    if task == "banking77":
        tr = _hf("mteb/banking77", None, "train")
        names = dict(zip(tr["label"], tr["label_text"]))
        assert sorted(names) == list(range(77))
        return [_intent(names[i]) for i in range(77)]
    if task == "clinc":
        names = _hf("clinc/clinc_oos", "plus", "train").features["intent"].names
        return [_intent(n) for n in names if n != "oos"]
    if task == "massive":
        return [_intent(n) for n in sorted(set(_hf("mteb/amazon_massive_intent", "en", "train")["label"]))]
    if task == "ledgar":
        return list(_hf("coastalcph/lex_glue", "ledgar", "train").features["label"].names)  # clause headings, Title Case
    if task == "trec":
        return [f"{TREC_COARSE[c]} {TREC_FINE.get(f, f)}" for c, f in (o.split(":") for o in _trec_order())]
    if task == "fewrel":
        p2n = json.loads((STORE / "data/fewrel/pid2name.json").read_text())
        return [p2n[r][0] for r in _fewrel()[1]]
    if task == "news20":
        return [NEWS20[n] for n in _news20("train")[2]]
    raise ValueError(task)


@cache
def _trec_order():
    tr = _hf("SetFit/TREC-QC", None, "train")
    names = dict(zip(tr["label"], tr["label_original"]))
    return [names[i] for i in sorted(names)]


def _raw(task, split):
    """Iterable of raw rows (dicts) of an official split, with a unified 'y' label index (-1 = unusable)."""
    path, name, *_ = SPECS[task]
    if task == "fewrel":
        rows, rels = _fewrel()
        perm = np.random.default_rng(3).permutation(len(rows))
        n_eval = int(round(0.2 * len(rows)))
        sel = perm[n_eval:] if split == "train" else perm[:n_eval]
        return [{"ex": rows[i][1], "y": rels.index(rows[i][0])} for i in sorted(sel)]
    if task == "news20":
        data, target, _ = _news20(split)
        return [{"text": x, "y": int(y)} for x, y in zip(data, target)]
    if task == "sick":
        import pandas as pd

        d = pd.read_parquet(STORE / "data/sick" / f"{split}.parquet")  # official parquet conversion (as the 3B run)
        return [{"a": a, "b": b, "y": int(y)} for a, b, y in zip(d["sentence_A"], d["sentence_B"], d["label"])]
    ds = _hf(path, name, split)
    if task == "clinc":
        names = ds.features["intent"].names
        keep = [n for n in names if n != "oos"]
        return [{"text": r["text"], "y": keep.index(names[r["intent"]]) if names[r["intent"]] != "oos" else -1}
                for r in ds]
    if task == "massive":
        order = sorted(set(_hf(path, name, "train")["label"]))
        return [{"text": r["text"], "y": order.index(r["label"])} for r in ds]
    if task == "trec":
        order = _trec_order()
        return [{"text": r["text"], "y": order.index(r["label_original"])} for r in ds]
    if task == "scitail":
        return [{"a": r["sentence1"], "b": r["sentence2"],
                 "y": {"entailment": 0, "neutral": 1}.get(r["gold_label"], -1)} for r in ds]
    return ds


def _cut(text, n=MAX_FIELD_TOKENS):
    tok = tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    return text if len(ids) <= n else tok.decode(ids[:n]).rstrip() + " ..."


def _fewrel_text(ex):
    toks = list(ex["tokens"])
    (hname, _, hpos), (tname, _, tpos) = ex["h"], ex["t"]
    marks = {}
    for span, (o, c) in ((hpos[0], ("[[", "]]")), (tpos[0], ("<<", ">>"))):
        marks[span[0]] = marks.get(span[0], "") + o + " "
        marks[span[-1] + 1000000] = " " + c
    out = []
    for i, w in enumerate(toks):
        out.append(marks.get(i, "") + w + marks.get(i + 1000000, ""))
    return " ".join(out), " ".join(toks[j] for j in hpos[0]), " ".join(toks[j] for j in tpos[0])


def _render(task, r):
    """(user-turn text, label index) or None when the row is unusable (no gold label, empty document)."""
    fam = family(task)
    t = TEMPLATES[fam]
    if task in ("mnli", "snli", "anli"):
        a, b, y = r["premise"], r["hypothesis"], r["label"]
    elif task == "qnli":
        a, b, y = r["question"], r["sentence"], r["label"]
    elif task in ("rte", "mrpc"):
        a, b, y = r["sentence1"], r["sentence2"], r["label"]
    elif task == "qqp":
        a, b, y = r["question1"], r["question2"], r["label"]
    elif task in ("sick", "scitail"):
        a, b, y = r["a"], r["b"], r["y"]
    elif task in ("banking77", "ledgar", "agnews"):
        a, y = r["text"], r["label"]
    elif task in ("cola", "sst2"):
        a, y = r["sentence"], r["label"]
    elif task == "dbpedia":
        a, y = f"{r['title']}. {r['content'].strip()}", r["label"]
    elif task in ("clinc", "massive", "trec", "news20"):
        a, y = r["text"], r["y"]
    elif task == "fewrel":
        a, b, c = _fewrel_text(r["ex"])
        return t.format(a=a, b=b, c=c), r["y"]
    else:
        raise ValueError(task)
    if y is None or y < 0:
        return None
    a = re.sub(r"\n{3,}", "\n\n", a.strip()) if task == "news20" else a
    if not str(a).strip():
        return None
    if fam in ("text", "nli"):
        a = _cut(a)
    return (t.format(a=a, b=b) if fam in ("nli", "pair") else t.format(a=a)), int(y)


@cache
def examples(task, split):
    """List of (user text, label index) for the usable rows of an official split (order preserved), cached on disk."""
    import pickle

    path = STORE / "cache" / f"examples_{task}_{split}.pkl"
    if path.exists():
        return pickle.load(path.open("rb"))
    raw = _raw(task, SPECS[task][2] if split == "train" else SPECS[task][3])
    out = [x for x in (_render(task, r) for r in raw) if x is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    pickle.dump(out, tmp.open("wb"))
    os.replace(tmp, path)
    return out


def prompt(user_text):
    return PREFIX + user_text + SUFFIX


# ----------------------------------------------------------------------------------------------- splits and pools
def eval_indices(task):
    """(val, test) indices into examples(task, 'eval'): seed-0 permutation, first 20 % val, rest test, capped."""
    n = len(examples(task, "eval"))
    perm = np.random.default_rng(0).permutation(n)
    n_val = int(round(0.2 * n))
    return perm[:n_val][:VAL_CAP], perm[n_val:][:TEST_CAP]


def train_indices(task):
    """(expert training indices, held-out reserve of max(50, 3 C) rows) into examples(task, 'train'). Seed 1."""
    n = len(examples(task, "train"))
    perm = np.random.default_rng(1).permutation(n)
    h = heldout_size(task)
    return perm[h:], perm[:h]


def expert_steps(task, batch=32, max_steps=1000, max_epochs=10):
    n = len(train_indices(task)[0])
    return int(min(max_steps, np.ceil(max_epochs * n / batch)))


def seen_stream(task, batch=32):
    """The expert's exact example stream: epochs of seed-2 permutations of its training indices, truncated to
    steps * batch (indices into examples(task, 'train'))."""
    train, _ = train_indices(task)
    need = expert_steps(task, batch) * batch
    rng = np.random.default_rng(2)
    out = []
    while len(out) < need:
        out.extend(train[rng.permutation(len(train))].tolist())
    return np.asarray(out[:need])


def pool_indices(task, per_task=50, seed=0, regime="train"):
    """Calibration pool rows of one task. regime 'train': distinct examples the expert actually trained on;
    'heldout': drawn from the reserve the expert never saw. per_task = 0 -> max(50, 3 C_t) (second-phase control)."""
    per = heldout_size(task) if per_task == 0 else per_task
    if regime == "heldout":
        _, held = train_indices(task)
        return np.sort(np.random.default_rng(seed).permutation(held)[:per])
    stream = seen_stream(task)
    seen = np.asarray(list(dict.fromkeys(stream.tolist())))  # distinct, in first-seen order
    rng = np.random.default_rng(1000 + TASKS20.index(task) + 7919 * seed)
    return np.sort(rng.choice(seen, size=min(per, len(seen)), replace=False))


def leak_check(tasks):
    """Dataset names in the prompt templates and label strings (must be none) and in the example texts (reported)."""
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in DATASET_NAMES) + r")\b", re.IGNORECASE)
    fixed = list(TEMPLATES.values()) + [PREFIX, SUFFIX]
    bad = [s for s in fixed if pat.search(s)] + [w for t in tasks for w in answers(t) if pat.search(w)]
    assert not bad, bad
    return {t: sum(bool(pat.search(x)) for x, _ in examples(t, "train")[:2000]) for t in tasks}
