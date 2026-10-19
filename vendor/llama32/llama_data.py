"""Data for the 20-task Llama-3.2-3B-Instruct LoRA bank (text port of the blind static merging framework).

Nested task lists (one bank, T8 subset T14 subset T20), from the text-extension plan
(reports/studies/llama32_text_protocol_2026-09-29.md):
  T8  = MNLI QNLI SNLI RTE SICK SciTail MRPC QQP          (hard core: six NLI + two paraphrase)
  T14 = T8 + CoLA WiC SST-2 AGNews DBpedia HellaSwag
  T20 = T14 + BoolQ COPA CommonsenseQA ARC-Challenge WSC CB

Prompts: one template per format family, never a dataset name. The user turn is wrapped in the Llama-3 chat
format by hand (the tokenizer's chat template injects today's date in a system block, which would make the
prompts depend on the run date). The assistant turn is empty: the answer is the next token.

Answers: every label is verbalised by ONE token (checked at load time), so p_t(y|x) over the allowed answers
H_t is the next-token softmax restricted to H_t: the exact candidate log-likelihood of the plan with |y| = 1,
one forward per prompt. Label sets with the same meaning share their words (the NLI tasks share yes/maybe/no),
different meanings use different words, so the union of the bank's answers (the "joint head") keeps
cross-family confusions measurable.

Splits (identical construction to run_vision_eval.py): the labelled evaluation source (official validation,
or test for AGNews/DBpedia) is permuted with seed 0; first 20 % = val (selection), remaining 80 % = test
(reporting), capped at VAL_CAP / TEST_CAP examples per task. The official train split is split into a 50-example
HELD-OUT reserve (never shown to the expert; held-out pool control) and the expert's training set.
"""
from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import numpy as np

STORE = Path("/shared/ssd_30T/molohio_llama32")
HF_HOME = STORE / "hf"
MODEL = "unsloth/Llama-3.2-3B-Instruct"  # byte-identical weights to meta-llama/Llama-3.2-3B-Instruct (sha256 checked)
PREFIX = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
SUFFIX = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
N_PREFIX = 5  # tokens of PREFIX: identical activations for every prompt (causal), excluded from READ

TASKS8 = ["mnli", "qnli", "snli", "rte", "sick", "scitail", "mrpc", "qqp"]
TASKS14 = TASKS8 + ["cola", "wic", "sst2", "agnews", "dbpedia", "hellaswag"]
TASKS20 = TASKS14 + ["boolq", "copa", "csqa", "arc", "wsc", "cb"]
TASKSETS = {8: TASKS8, 14: TASKS14, 20: TASKS20}
VAL_CAP, TEST_CAP, HELDOUT = 250, 1000, 50
MAX_FIELD_TOKENS = 256  # long fields (passages, articles, contexts) are cut to this many tokens

NLI3 = ["yes", "maybe", "no"]            # entailment, neutral, contradiction
NLI2 = ["yes", "no"]                     # entailment, not entailment
PARA = ["different", "same"]             # not paraphrase, paraphrase
BOOL = ["false", "true"]
LETTERS = ["A", "B", "C", "D", "E"]

# task -> (hf path, config, train split, eval split, family, answers in label-index order)
SPECS = {
    "mnli": ("nyu-mll/glue", "mnli", "train", "validation_matched", "nli", NLI3),
    "qnli": ("nyu-mll/glue", "qnli", "train", "validation", "nli", NLI2),
    "snli": ("stanfordnlp/snli", None, "train", "validation", "nli", NLI3),
    "rte": ("nyu-mll/glue", "rte", "train", "validation", "nli", NLI2),
    "sick": ("RobZamp/sick", "parquet", "train", "validation", "nli", NLI3),
    "scitail": ("allenai/scitail", "snli_format", "train", "validation", "nli", NLI2),
    "mrpc": ("nyu-mll/glue", "mrpc", "train", "validation", "pair", PARA),
    "qqp": ("nyu-mll/glue", "qqp", "train", "validation", "pair", PARA),
    "cola": ("nyu-mll/glue", "cola", "train", "validation", "text", ["incorrect", "correct"]),
    "wic": ("aps/super_glue", "wic", "train", "validation", "question", BOOL),
    "sst2": ("nyu-mll/glue", "sst2", "train", "validation", "text", ["negative", "positive"]),
    "agnews": ("fancyzhx/ag_news", None, "train", "test", "text", ["world", "sports", "business", "technology"]),
    "dbpedia": ("fancyzhx/dbpedia_14", None, "train", "test", "text",
                ["company", "school", "artist", "athlete", "official", "vehicle", "building", "nature", "town",
                 "animal", "plant", "album", "film", "book"]),
    "hellaswag": ("Rowan/hellaswag", None, "train", "validation", "choice", LETTERS[:4]),
    "boolq": ("google/boolq", None, "train", "validation", "question", BOOL),
    "copa": ("aps/super_glue", "copa", "train", "validation", "choice", LETTERS[:2]),
    "csqa": ("tau/commonsense_qa", None, "train", "validation", "choice", LETTERS[:5]),
    "arc": ("allenai/ai2_arc", "ARC-Challenge", "train", "validation", "choice", LETTERS[:4]),
    "wsc": ("aps/super_glue", "wsc.fixed", "train", "validation", "question", BOOL),
    "cb": ("aps/super_glue", "cb", "train", "validation", "nli", ["yes", "no", "maybe"]),  # CB: 0 ent, 1 contr, 2 neutral
}
DATASET_NAMES = ["MNLI", "SNLI", "QNLI", "RTE", "SICK", "SciTail", "MRPC", "QQP", "CoLA", "WiC", "SST", "AGNews",
                 "AG News", "DBpedia", "HellaSwag", "BoolQ", "COPA", "CommonsenseQA", "ARC", "WSC", "CB", "GLUE",
                 "SuperGLUE"]


def answers(task):
    return SPECS[task][5]


def family(task):
    return SPECS[task][4]


def _env():
    import os

    os.environ.setdefault("HF_HOME", str(HF_HOME))


@cache
def tokenizer():
    _env()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    assert len(tok.encode(PREFIX, add_special_tokens=False)) == N_PREFIX
    return tok


@cache
def answer_id(word):
    ids = tokenizer().encode(word, add_special_tokens=False)
    assert len(ids) == 1, (word, ids)
    return ids[0]


def union(tasks):
    """The bank's answer vocabulary U (ordered, unique words) and, per task, the U-indices of H_t."""
    words = []
    for t in tasks:
        for w in answers(t):
            if w not in words:
                words.append(w)
    maps = {t: np.asarray([words.index(w) for w in answers(t)]) for t in tasks}
    return words, np.asarray([answer_id(w) for w in words]), maps


# ----------------------------------------------------------------------------------------------- raw loading
@cache
def _raw(task, split):
    _env()
    from datasets import load_dataset

    path, name, *_ = SPECS[task]
    if name == "parquet":  # RobZamp/sick is a script dataset; its official parquet conversion (refs/convert/parquet,
        # copied once to STORE/data/sick) has the same rows
        return load_dataset("parquet", data_files={split: str(STORE / "data/sick" / f"{split}.parquet")})[split]
    return load_dataset(path, name, split=split)


def _cut(text, n=MAX_FIELD_TOKENS):
    tok = tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    return text if len(ids) <= n else tok.decode(ids[:n]).rstrip() + " ..."


def _hellaswag_clean(text):
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ").strip()


def _mc(question, choices):
    lines = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"Question:\n{question}\n\nChoices:\n{lines}\n\nAnswer:"


def _render(task, r):
    """(user-turn text, label index) or None when the row is unusable (no gold label, non-4-choice ARC)."""
    fam = family(task)
    if task in ("mnli", "snli", "cb"):
        p, h, y = r["premise"], r["hypothesis"], r["label"]
    elif task == "qnli":
        p, h, y = r["question"], r["sentence"], r["label"]
    elif task == "rte":
        p, h, y = r["sentence1"], r["sentence2"], r["label"]
    elif task == "sick":
        p, h, y = r["sentence_A"], r["sentence_B"], r["label"]
    elif task == "scitail":
        p, h, y = r["sentence1"], r["sentence2"], {"entailment": 0, "neutral": 1}.get(r["gold_label"], -1)
    if fam == "nli":
        return (f"Premise:\n{p}\n\nHypothesis:\n{h}\n\nAnswer:", y) if y >= 0 else None
    if fam == "pair":
        a, b = (r["sentence1"], r["sentence2"]) if task == "mrpc" else (r["question1"], r["question2"])
        return f"Sentence 1:\n{a}\n\nSentence 2:\n{b}\n\nAnswer:", r["label"]
    if fam == "text":
        if task == "dbpedia":
            x = f"{r['title']}. {r['content'].strip()}"
        elif task == "agnews":
            x = r["text"]
        else:
            x = r["sentence"]
        return (f"Text:\n{_cut(x)}\n\nAnswer:", r["label"]) if r["label"] >= 0 else None
    if fam == "question":
        if task == "boolq":
            passage, q, y = r["passage"], r["question"].strip().capitalize() + "?", int(r["answer"])
        elif task == "wsc":
            passage, y = r["text"], r["label"]
            q = f'Does the pronoun "{r["span2_text"]}" refer to "{r["span1_text"]}"?'
        else:  # wic
            passage, y = f"{r['sentence1']}\n{r['sentence2']}", r["label"]
            q = f'Is the word "{r["word"]}" used with the same meaning in both sentences?'
        return (f"Passage:\n{_cut(passage)}\n\nQuestion:\n{q}\n\nAnswer:", y) if y >= 0 else None
    # multiple choice
    if task == "hellaswag":
        if r["label"] in ("", None):
            return None
        return _mc(_cut(_hellaswag_clean(r["ctx"])), [_hellaswag_clean(e) for e in r["endings"]]), int(r["label"])
    if task == "copa":
        what = "What was the cause?" if r["question"] == "cause" else "What happened as a result?"
        return (_mc(f"{r['premise']} {what}", [r["choice1"], r["choice2"]]), r["label"]) if r["label"] >= 0 else None
    if task == "csqa":
        labels = list(r["choices"]["label"])
        if not r["answerKey"] or labels != LETTERS[:5]:
            return None
        return _mc(r["question"], r["choices"]["text"]), labels.index(r["answerKey"])
    if task == "arc":
        labels = list(r["choices"]["label"])
        if len(labels) != 4 or r["answerKey"] not in labels:
            return None
        return _mc(r["question"], r["choices"]["text"]), labels.index(r["answerKey"])
    raise ValueError(task)


@cache
def examples(task, split):
    """List of (user text, label index) for the usable rows of an official split (order preserved).
    Cached on disk (rendering DBpedia's 560k rows takes minutes)."""
    import os
    import pickle

    path = STORE / "cache" / f"examples_{task}_{split}.pkl"
    if path.exists():
        return pickle.load(path.open("rb"))
    raw = _raw(task, SPECS[task][2] if split == "train" else SPECS[task][3])
    out = []
    for r in raw:
        x = _render(task, r)
        if x is not None:
            out.append(x)
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
    """(expert training indices, held-out reserve) into examples(task, 'train'). Seed 1 permutation."""
    n = len(examples(task, "train"))
    perm = np.random.default_rng(1).permutation(n)
    return perm[HELDOUT:], perm[:HELDOUT]


def expert_steps(task, batch=32, max_steps=1000, max_epochs=10):
    n = len(train_indices(task)[0])
    return int(min(max_steps, np.ceil(max_epochs * n / batch)))


def seen_stream(task, batch=32):
    """The expert's exact example stream: epochs of seed-2 permutations of its training indices, truncated to
    steps * batch. Returns the stream (indices into examples(task, 'train'))."""
    train, _ = train_indices(task)
    need = expert_steps(task, batch) * batch
    rng = np.random.default_rng(2)
    out = []
    while len(out) < need:
        out.extend(train[rng.permutation(len(train))].tolist())
    return np.asarray(out[:need])


def pool_indices(task, per_task=50, seed=0, regime="train"):
    """Calibration pool rows of one task. regime 'train': distinct examples the expert actually trained on
    (the TRAIN409 analogue); 'heldout': the 50-example reserve the expert never saw."""
    if regime == "heldout":
        _, held = train_indices(task)
        return np.sort(np.random.default_rng(seed).permutation(held)[:per_task])
    stream = seen_stream(task)
    seen = np.asarray(list(dict.fromkeys(stream.tolist())))  # distinct, in first-seen order
    rng = np.random.default_rng(1000 + TASKS20.index(task) + 7919 * seed)
    return np.sort(rng.choice(seen, size=min(per_task, len(seen)), replace=False))


def leak_check(tasks):
    """Dataset names in the prompt templates (must be none) and in the example texts (reported, allowed)."""
    pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in DATASET_NAMES) + r")\b")
    templates = ["Premise:", "Hypothesis:", "Sentence 1:", "Sentence 2:", "Text:", "Passage:", "Question:", "Choices:",
                 "Answer:", "What was the cause?", "What happened as a result?", "Does the pronoun", "Is the word",
                 "used with the same meaning in both sentences?", PREFIX, SUFFIX]
    assert not any(pat.search(s) for s in templates)
    report = {}
    for t in tasks:
        hits = sum(bool(pat.search(x)) for x, _ in examples(t, "train")[:2000])
        report[t] = hits
    return report
