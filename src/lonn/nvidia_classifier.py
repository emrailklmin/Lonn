"""Profile a prompt with NVIDIA's prompt task and complexity classifier.

`nvidia/prompt-task-and-complexity-classifier` is DeBERTa-v3-base with one
linear head per output, trained on 4,024 human-annotated prompts for routing.
It runs locally (no API calls) and returns, for one prompt:

- a task type distribution over 12 labels (Code Generation, Open QA, ...);
- six complexity dimensions in [0, 1] (number of few-shots in [0, 5]);
- NVIDIA's overall complexity score, a fixed weighted sum of those dimensions.

The model is loaded lazily on first use and reused. Prompts longer than the
512-token window keep their start and end, as in `complexity.py`.
Licensed under the NVIDIA Open Model License.
"""

from dataclasses import dataclass
import json
import logging
import os

logger = logging.getLogger(__name__)

REPO = "nvidia/prompt-task-and-complexity-classifier"
BACKBONE = "microsoft/deberta-v3-base"
MAX_TOKENS = 512
HEAD_SHARE = 0.35  # Share of an over-long prompt kept from its start.

# NVIDIA's overall complexity weights, from the model card.
OVERALL_WEIGHTS = {
    "creativity": 0.35,
    "reasoning": 0.25,
    "constraints": 0.15,
    "domain_knowledge": 0.15,
    "contextual_knowledge": 0.05,
    "few_shots": 0.05,
}

# Model output names (config.json `target_sizes` keys) -> PromptProfile fields.
DIMENSIONS = {
    "creativity_scope": "creativity",
    "reasoning": "reasoning",
    "contextual_knowledge": "contextual_knowledge",
    "number_of_few_shots": "few_shots",
    "domain_knowledge": "domain_knowledge",
    "constraint_ct": "constraints",
}


@dataclass
class PromptProfile:
    """The classifier's view of one prompt."""

    task_type: str
    task_probabilities: dict[str, float]
    creativity: float
    reasoning: float
    contextual_knowledge: float
    few_shots: float
    domain_knowledge: float
    constraints: float
    complexity: float


def _build_model(config: dict):
    """DeBERTa backbone plus heads named as in the published weights."""

    import torch.nn as nn
    from transformers import AutoConfig, AutoModel

    class Head(nn.Module):
        def __init__(self, size_in, size_out):
            super().__init__()
            self.fc = nn.Linear(size_in, size_out)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_config(AutoConfig.from_pretrained(BACKBONE))
            hidden = self.backbone.config.hidden_size
            for i, size in enumerate(config["target_sizes"].values()):
                self.add_module(f"head_{i}", Head(hidden, size))

        def forward(self, input_ids, attention_mask):
            hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return [getattr(self, f"head_{i}").fc(pooled) for i in range(len(config["target_sizes"]))]

    return Classifier()


class NvidiaPromptClassifier:
    """Lazily loaded NVIDIA prompt classifier. Pass `device` to override the default."""

    def __init__(self, device: str | None = None):
        self._device = device
        self._model = None

    @property
    def device(self) -> str | None:
        """Device the model runs on; chosen at load time unless given."""
        return self._device

    def load(self) -> "NvidiaPromptClassifier":
        """Load the model now instead of on the first prompt."""
        if self._model is None:
            self._load()
        return self

    def _load(self):
        os.environ.setdefault("USE_TF", "0")  # Avoids a known TensorFlow import hang.

        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        with open(hf_hub_download(REPO, "config.json")) as f:
            self._config = json.load(f)
        self._tokenizer = AutoTokenizer.from_pretrained(REPO)
        model = _build_model(self._config)
        model.load_state_dict(load_file(hf_hub_download(REPO, "model.safetensors")), strict=True)

        if self._device is None:
            self._device = "mps" if torch.backends.mps.is_available() else (
                "cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device).eval()
        self._torch = torch

        targets = list(self._config["target_sizes"])
        self._task_head = targets.index("task_type")
        self._task_labels = [self._config["task_type_map"][str(i)]
                             for i in range(self._config["target_sizes"]["task_type"])]
        self._dimension_heads = {targets.index(name): name for name in DIMENSIONS}

    def _fit(self, prompt: str) -> str:
        """Shorten an over-long prompt to its start and end, like `complexity.py`."""

        ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        budget = MAX_TOKENS - 8  # Room for special tokens and the " ... " joint.
        if len(ids) <= budget:
            return prompt
        head = int(budget * HEAD_SHARE)
        decode = lambda part: self._tokenizer.decode(part, clean_up_tokenization_spaces=False)
        return f"{decode(ids[:head])} ... {decode(ids[-(budget - head):])}"

    def classify(self, prompt: str) -> PromptProfile:
        return self.classify_batch([prompt])[0]

    def classify_batch(self, prompts: list[str]) -> list[PromptProfile]:
        self.load()
        torch = self._torch
        batch = self._tokenizer([self._fit(p) for p in prompts], padding=True, truncation=True,
                                max_length=MAX_TOKENS, return_tensors="pt").to(self._device)
        with torch.no_grad():
            logits = self._model(batch["input_ids"], batch["attention_mask"])
        task = torch.softmax(logits[self._task_head].float(), -1).cpu().tolist()

        dimensions = {}
        for index, name in self._dimension_heads.items():
            probs = torch.softmax(logits[index].float(), -1).cpu()
            weights = torch.tensor(self._config["weights_map"][name], dtype=probs.dtype)
            dimensions[DIMENSIONS[name]] = ((probs * weights).sum(-1) / self._config["divisor_map"][name]).tolist()

        profiles = []
        for i, probs in enumerate(task):
            values = {field: column[i] for field, column in dimensions.items()}
            distribution = dict(zip(self._task_labels, probs))
            profiles.append(PromptProfile(
                task_type=max(distribution, key=distribution.get),
                task_probabilities=distribution,
                complexity=sum(OVERALL_WEIGHTS[k] * values[k] for k in OVERALL_WEIGHTS),
                **values,
            ))
        return profiles
