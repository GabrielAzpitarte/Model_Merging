"""Zero-shot prompt templates for the 8-task vision benchmark (vendored from the ICA project).

Reproduced from the KnOTS release (``dataset/templates.py``), which takes them
from the task-vectors line of work (Ilharco et al.), which takes them from
CLIP's own zero-shot evaluation.  They are vendored here rather than imported
because reproducing a published number means using *that paper's* prompts:
changing a single template moves zero-shot accuracy by whole points, and a
mismatch would be indistinguishable from a bug in the merge.
"""

from __future__ import annotations

__all__ = ["TEMPLATES", "get_templates"]

TEMPLATES: dict[str, list] = {
    "cars": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a photo of the {c}.",
        lambda c: f"a photo of my {c}.",
        lambda c: f"i love my {c}!",
        lambda c: f"a photo of my dirty {c}.",
        lambda c: f"a photo of my clean {c}.",
        lambda c: f"a photo of my new {c}.",
        lambda c: f"a photo of my old {c}.",
    ],
    "dtd": [
        lambda c: f"a photo of a {c} texture.",
        lambda c: f"a photo of a {c} pattern.",
        lambda c: f"a photo of a {c} thing.",
        lambda c: f"a photo of a {c} object.",
        lambda c: f"a photo of the {c} texture.",
        lambda c: f"a photo of the {c} pattern.",
        lambda c: f"a photo of the {c} thing.",
        lambda c: f"a photo of the {c} object.",
    ],
    "gtsrb": [
        lambda c: f'a zoomed in photo of a "{c}" traffic sign.',
        lambda c: f'a centered photo of a "{c}" traffic sign.',
        lambda c: f'a close up photo of a "{c}" traffic sign.',
    ],
    "mnist": [lambda c: f'a photo of the number: "{c}".'],
    "eurosat": [
        lambda c: f"a centered satellite photo of {c}.",
        lambda c: f"a centered satellite photo of a {c}.",
        lambda c: f"a centered satellite photo of the {c}.",
    ],
    "resisc45": [
        lambda c: f"satellite imagery of {c}.",
        lambda c: f"aerial imagery of {c}.",
        lambda c: f"satellite photo of {c}.",
        lambda c: f"aerial photo of {c}.",
        lambda c: f"satellite view of {c}.",
        lambda c: f"aerial view of {c}.",
        lambda c: f"satellite imagery of a {c}.",
        lambda c: f"aerial imagery of a {c}.",
        lambda c: f"satellite photo of a {c}.",
        lambda c: f"aerial photo of a {c}.",
        lambda c: f"satellite view of a {c}.",
        lambda c: f"aerial view of a {c}.",
        lambda c: f"satellite imagery of the {c}.",
        lambda c: f"aerial imagery of the {c}.",
        lambda c: f"satellite photo of the {c}.",
        lambda c: f"aerial photo of the {c}.",
        lambda c: f"satellite view of the {c}.",
        lambda c: f"aerial view of the {c}.",
    ],
    "sun397": [
        lambda c: f"a photo of a {c}.",
        lambda c: f"a photo of the {c}.",
    ],
    "svhn": [lambda c: f'a photo of the number: "{c}".'],
}


def get_templates(task: str) -> list:
    if task not in TEMPLATES:
        raise KeyError(f"No templates for {task!r}; known: {sorted(TEMPLATES)}")
    return TEMPLATES[task]
