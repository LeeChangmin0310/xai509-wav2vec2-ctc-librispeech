"""Small JSON-serializable word n-gram LM compatible with pyctcdecode."""

import json
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from pyctcdecode.language_model import AbstractLanguageModel


class SimpleWordNGramLanguageModel(AbstractLanguageModel):
    """Add-k word n-gram model for validation-tuned shallow fusion."""

    def __init__(
        self,
        order: int,
        vocabulary: List[str],
        counts: Dict[int, Dict[Tuple[str, ...], Counter]],
        alpha: float = 0.5,
        beta: float = 1.5,
        add_k: float = 0.1,
    ):
        self._order = order
        self.vocabulary = set(vocabulary)
        self.counts = counts
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.add_k = float(add_k)
        self._prefixes = {
            word[:index]
            for word in self.vocabulary
            for index in range(1, len(word) + 1)
        }

    @property
    def order(self) -> int:
        return self._order

    def get_start_state(self) -> Tuple[str, ...]:
        return tuple(["<s>"] * max(0, self.order - 1))

    def score_partial_token(self, partial_token: str) -> float:
        if not partial_token or partial_token in self._prefixes:
            return 0.0
        return -5.0 * self.alpha

    def _log_probability(self, state: Tuple[str, ...], word: str) -> float:
        vocabulary_size = len(self.vocabulary) + 1
        max_context = min(len(state), self.order - 1)
        for context_size in range(max_context, -1, -1):
            context = state[-context_size:] if context_size else ()
            next_counts = self.counts.get(context_size + 1, {}).get(context)
            if next_counts:
                numerator = next_counts.get(word, 0) + self.add_k
                denominator = sum(next_counts.values()) + self.add_k * vocabulary_size
                return math.log(numerator / denominator)
        return -math.log(vocabulary_size)

    def score(
        self,
        prev_state: Tuple[str, ...],
        word: str,
        is_last_word: bool = False,
    ):
        lm_score = self._log_probability(prev_state, word)
        next_state = tuple(
            (list(prev_state) + [word])[-max(0, self.order - 1) :]
        )
        if is_last_word:
            lm_score += self._log_probability(next_state, "</s>")
        return self.alpha * lm_score + self.beta, next_state

    def reset_params(self, **params) -> None:
        if "alpha" in params and params["alpha"] is not None:
            self.alpha = float(params["alpha"])
        if "beta" in params and params["beta"] is not None:
            self.beta = float(params["beta"])

    def save_to_dir(self, filepath: str) -> None:
        self.save(filepath)

    @classmethod
    def load_from_dir(cls, filepath: str):
        return cls.load(filepath)

    def save(self, filepath: str) -> None:
        serializable_counts = {}
        for ngram_order, context_counts in self.counts.items():
            serializable_counts[str(ngram_order)] = {
                "\t".join(context): dict(next_counts)
                for context, next_counts in context_counts.items()
            }
        payload = {
            "order": self.order,
            "vocabulary": sorted(self.vocabulary),
            "counts": serializable_counts,
            "alpha": self.alpha,
            "beta": self.beta,
            "add_k": self.add_k,
        }
        with open(filepath, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, sort_keys=True)
            output_file.write("\n")

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
        counts = {}
        for ngram_order, context_counts in payload["counts"].items():
            counts[int(ngram_order)] = {
                tuple(context.split("\t")) if context else (): Counter(next_counts)
                for context, next_counts in context_counts.items()
            }
        return cls(
            order=payload["order"],
            vocabulary=payload["vocabulary"],
            counts=counts,
            alpha=payload.get("alpha", 0.5),
            beta=payload.get("beta", 1.5),
            add_k=payload.get("add_k", 0.1),
        )

    @classmethod
    def fit(
        cls,
        sentences: Iterable[str],
        order: int = 3,
        add_k: float = 0.1,
    ):
        counts = {
            ngram_order: defaultdict(Counter)
            for ngram_order in range(1, order + 1)
        }
        vocabulary = set()
        for sentence in sentences:
            words = sentence.split()
            vocabulary.update(words)
            padded = ["<s>"] * max(0, order - 1) + words + ["</s>"]
            for index in range(order - 1, len(padded)):
                word = padded[index]
                for ngram_order in range(1, order + 1):
                    context_size = ngram_order - 1
                    context = (
                        tuple(padded[index - context_size : index])
                        if context_size
                        else ()
                    )
                    counts[ngram_order][context][word] += 1
        return cls(order, sorted(vocabulary), counts, add_k=add_k)
