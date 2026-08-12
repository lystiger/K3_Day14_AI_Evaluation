"""Alternative generators for `domain_assistant.py` when OpenAI is unavailable.

Why this file exists
--------------------
The lab's default generator is OpenAI (`OPENAI_API_KEY` in `.env`). On this
machine that key returns ``429 insufficient_quota``, so the default run cannot
finish. ``domain_assistant.py`` is the system under evaluation and is **not
modified** here: it already exposes a public hook,
``generate_actual_answers(..., generator=...)``, which accepts any object
implementing ``TextGenerator.generate(prompt) -> str``.

Everything else in the pipeline is untouched — same corpus, same BM25 retriever,
same prompt built by ``_build_prompt``, same ``top_k``:

    question → BM25 retrieval → retrieved chunks → <generator> → actual answer

Both generators below only ever see the prompt string. They never read
``expected_answer`` or gold evidence, so no gold data leaks into the benchmark.

Backends
--------
``--backend groq`` (default)
    A real LLM (Groq, OpenAI-compatible endpoint), temperature 0. This is the
    closest available substitute for the OpenAI baseline and is what the
    reported benchmark uses.

``--backend extractive``
    A deterministic, offline sentence selector that needs no API at all. Useful
    as a zero-cost smoke test and as a contrast case: it can only copy sentences
    the retriever supplied.

The artifact records the backend under ``agent.model``, so runs can never be
confused in a report.

Usage:
    python fallback_generators.py                      # groq
    python fallback_generators.py --backend extractive # offline
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from domain_assistant import _tokenize, generate_actual_answers

load_dotenv(Path(__file__).resolve().with_name(".env"))

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
CONTEXT_HEADER_RE = re.compile(r"^\[Context \d+ \| (?P<source>[^\]]+)\]$")


class GroqGenerator:
    """Groq chat-completions generator with the same contract as OpenAIGenerator.

    Groq exposes an OpenAI-compatible endpoint, so the official ``openai`` SDK
    is reused with a different ``base_url``. ``responses.create`` is not
    available there, hence ``chat.completions.create``.
    """

    def __init__(self, max_output_tokens: int = 300) -> None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing from .env")
        self.client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=self.max_output_tokens,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("Groq returned an empty answer")
        return answer


class ExtractiveGenerator:
    """Select the prompt sentences that best match the question.

    A deliberately simple, non-neural baseline: it can only copy sentences that
    the retriever supplied, so it cannot hallucinate outside the corpus, but it
    also cannot synthesise, refuse, or reason about conditions.
    """

    model = "extractive-offline"

    def __init__(self, max_sentences: int = 4) -> None:
        self.max_sentences = max_sentences

    def generate(self, prompt: str) -> str:
        question, sentences = self._parse_prompt(prompt)
        query_tokens = set(_tokenize(question))
        if not sentences:
            return "The retrieved documents do not contain an answer to this question."

        scored: list[tuple[float, int, str]] = []
        for order, sentence in enumerate(sentences):
            tokens = set(_tokenize(sentence))
            if not tokens:
                continue
            # Normalising by sentence length keeps very long sentences from
            # winning purely because they contain more words.
            scored.append(
                (len(tokens & query_tokens) / (len(tokens) ** 0.5), order, sentence)
            )

        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = [item for item in scored[: self.max_sentences] if item[0] > 0]
        if not chosen:
            return "The retrieved documents do not contain an answer to this question."
        chosen.sort(key=lambda item: item[1])  # restore document order
        return " ".join(sentence for _, _, sentence in chosen)

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[str, list[str]]:
        """Split the assistant prompt into the question and context sentences."""
        question_part = prompt.split("Question:", 1)[-1]
        question, _, contexts_part = question_part.partition("Retrieved contexts:")
        contexts_part = contexts_part.rsplit("Answer:", 1)[0]

        sentences: list[str] = []
        for line in contexts_part.splitlines():
            line = line.strip()
            if not line or CONTEXT_HEADER_RE.match(line):
                continue
            sentences.extend(
                part.strip() for part in SENTENCE_RE.split(line) if part.strip()
            )
        return question.strip(), sentences


BACKENDS = {"groq": GroqGenerator, "extractive": ExtractiveGenerator}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="groq")
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/student_services"))
    parser.add_argument("--dataset", type=Path, default=Path("golden_dataset.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/actual_answers.json")
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        artifact = generate_actual_answers(
            args.dataset,
            args.corpus_dir,
            generator=BACKENDS[args.backend](),
            top_k=args.top_k,
            progress=lambda message: print(message, flush=True),
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}")
        return 2

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(artifact['answers'])} actual answers: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
