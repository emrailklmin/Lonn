"""Classify a prompt into task types (issue #19, docs/classifier_bakeoff.md).

A frozen sentence embedder (BAAI/bge-small-en-v1.5, MIT) plus a linear layer
trained on the task data (scripts/train_task_classifier.py). The layer lives in
config/task_classifier.json, which pins the embedder revision. The embedder is
loaded lazily on first use and runs on CPU.

With a complexity head (``complexity_path``, from scripts/train_complexity.py), a
second linear layer on the same embedding predicts easy / medium / hard, and the
signal's complexity is its expected position (easy 0, medium 0.5, hard 1).
Otherwise complexity is None. Production loads no head: predicted complexity did
not improve routing (docs/oracle_gap.md).

The classifier knows nothing about routing: it returns a TaskSignal.
"""

import json
import os
from pathlib import Path

from .schemas import TaskSignal

DEFAULT_TASK_CLASSIFIER = Path(__file__).resolve().parents[2] / 'config' / 'task_classifier.json'
LEVEL_POSITION = {'easy': 0.0, 'medium': 0.5, 'hard': 1.0}
HEAD_SHARE = 0.35  # Over-long prompts keep this share of the window from their start, the rest from the end.


def head_tail(ids: list[int], budget: int) -> list[int]:
    if len(ids) <= budget:
        return ids
    head = int(budget * HEAD_SHARE)
    return ids[:head] + ids[-(budget - head):]


class Embedder:
    """CLS-pooled, L2-normalized embeddings; long prompts keep their start and end."""

    def __init__(self, repo: str, max_len: int = 512, revision: str | None = None,
                 cache_dir=None, device: str = 'cpu'):
        os.environ.setdefault('USE_TF', '0')
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(repo, revision=revision, cache_dir=cache_dir).to(device).eval()
        self.max_len, self.device = max_len, device

    def embed(self, texts: list[str], batch_size: int = 32, progress: str = ''):
        import numpy as np
        torch = self.torch
        # verbose=False: over-long prompts are expected here; head_tail shortens them.
        ids = [head_tail(self.tok(t, add_special_tokens=False, verbose=False)['input_ids'], self.max_len - 2)
               for t in texts]
        order = np.argsort([len(i) for i in ids])
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                batch = [ids[i] for i in order[start:start + batch_size]]
                width = max(len(b) for b in batch) + 2
                input_ids = torch.full((len(batch), width), pad)
                mask = torch.zeros((len(batch), width), dtype=torch.long)
                for row, b in enumerate(batch):
                    seq = [cls] + b + [sep]
                    input_ids[row, :len(seq)] = torch.tensor(seq)
                    mask[row, :len(seq)] = 1
                hidden = self.model(input_ids=input_ids.to(self.device),
                                    attention_mask=mask.to(self.device)).last_hidden_state
                vectors = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1).cpu().numpy()
                out[order[start:start + batch_size]] = vectors
                if progress and start % (batch_size * 100) == 0:
                    print(f'    {progress}: {start}/{len(texts)}', flush=True)
        return out


class TaskClassifier:
    """Prompt -> TaskSignal. Pass ``embed`` (texts -> unit vectors) to skip the embedder in tests."""

    def __init__(self, path: str | Path = DEFAULT_TASK_CLASSIFIER, embed=None,
                 complexity_path: str | Path | None = None):
        config = json.loads(Path(path).read_text())
        self.config = config
        self.task_types: list[str] = config['task_types']
        self.weight = config['weight']  # one row per task type
        self.bias = config['bias']
        self.scale = float(config['logit_scale'])
        if len(self.weight) != len(self.task_types) or len(self.bias) != len(self.task_types):
            raise ValueError('Task classifier weights do not match its task types')
        self.complexity = None
        if complexity_path is not None:
            head = json.loads(Path(complexity_path).read_text())
            if head['embedder'] != config['embedder']:
                raise ValueError('The complexity head was trained on a different embedder')
            if len(head['weight']) != len(head['levels']) or not set(head['levels']) <= set(LEVEL_POSITION):
                raise ValueError('Complexity head weights do not match its levels')
            self.complexity = head
        self._embed = embed

    def _embedder(self):
        if self._embed is None:
            e = self.config['embedder']
            self._embed = Embedder(e['repo'], e['max_len'], e['revision']).embed
        return self._embed

    @staticmethod
    def _softmax(vectors, weight, bias, scale):
        import numpy as np
        logits = (vectors @ np.asarray(weight, dtype=np.float64).T + np.asarray(bias)) * scale
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)

    def _vectors(self, prompts):
        import numpy as np
        return np.asarray(self._embedder()(prompts), dtype=np.float64)

    def probabilities(self, prompts: list[str]) -> list[dict[str, float]]:
        probs = self._softmax(self._vectors(prompts), self.weight, self.bias, self.scale)
        return [dict(zip(self.task_types, map(float, row))) for row in probs]

    def signals(self, prompts: list[str]) -> list[TaskSignal]:
        """Task-type probabilities and, with a complexity head, complexity, from one embedding."""
        vectors = self._vectors(prompts)
        tasks = self._softmax(vectors, self.weight, self.bias, self.scale)
        levels = None
        if self.complexity:
            head = self.complexity
            probs = self._softmax(vectors, head['weight'], head['bias'], float(head['logit_scale']))
            positions = [LEVEL_POSITION[level] for level in head['levels']]
            levels = [float(min(1.0, max(0.0, sum(p * x for p, x in zip(row, positions))))) for row in probs]
        return [TaskSignal(dict(zip(self.task_types, map(float, row))), None if levels is None else levels[i])
                for i, row in enumerate(tasks)]

    def classify(self, prompt: str) -> TaskSignal:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Prompt must be a non-empty string')
        return self.signals([prompt])[0]
