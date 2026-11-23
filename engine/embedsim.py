"""문장 임베딩 유사도 (신경망) — textsim.py의 단어 가중 유사도 옆에 붙이는 선택 층.

왜 넣나
  textsim은 개념 사전(관세, 희토류, 제재 ...)에 있는 단어만 '같은 뜻'으로 묶는다. 사전에 없는 표현
  ("핵심 소재 반출 제한" ≈ "수출통제", "칩 장비 판매 금지" ≈ "반도체 장비 수출통제")은 놓친다.
  문장 임베딩 모델은 대량의 문장으로 사전학습된 신경망이라, 단어가 달라도 뜻이 가까우면 벡터가 가깝다.

왜 기본값이 꺼져 있나 (thresholds.yaml: analog.embed_weight = 0.0)
  1) 임베딩은 "관세 유예"와 "관세 부과"를 가깝게 본다(주제가 같으므로). 방향은 여기서 판단하지 않고,
     analog.py의 방향 필터가 계속 맡는다. 이 층은 '같은 주제인가'만 더한다.
  2) 효과를 판정할 기준이 아직 없다. agents/analog_eval.py의 사람 정답(data/library/analog_labels.yaml)이
     2026-11-23 기준 전부 비어 있어 hit@3을 계산할 수 없다. 라벨을 채운 뒤
     `python -m agents.analog_eval --embed 0.3` 으로 켠 것과 끈 것을 비교하고 나서 켤지 정한다.
  3) 모델 파일이 필요하다(처음 한 번 HuggingFace에서 받음). 설치돼 있지 않으면 단어 유사도만 쓰고
     메모를 남긴다 — 조용히 다른 숫자를 만들지 않는다.

결합
  문장 점수 = (1 - embed_weight) × 단어 유사도 + embed_weight × 임베딩 유사도
  임베딩 코사인은 음수가 될 수 있으므로 0 아래는 0으로 자른다(단어 유사도와 같은 [0, 1] 범위).

결정성
  같은 모델·같은 문장이면 같은 벡터가 나온다(추론만 하고 학습하지 않는다). 모델 이름은 결과 메모에 남긴다.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from engine.textsim import event_text

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"   # 한국어·영어 같이 쓰는 기사에 맞춘 다국어 모델(약 470MB)

Encoder = Callable[[Sequence[str]], np.ndarray]


class EmbeddingUnavailable(RuntimeError):
    """임베딩 백엔드를 쓸 수 없음(패키지 없음, 모델 다운로드 실패 등)."""


def sentence_transformers_encoder(model: str = DEFAULT_MODEL) -> Encoder:
    """sentence-transformers 모델로 인코더를 만든다. 패키지·모델이 없으면 EmbeddingUnavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # 선택 의존성
        raise EmbeddingUnavailable("sentence-transformers 미설치 (pip install sentence-transformers)") from e
    try:
        m = SentenceTransformer(model)
    except Exception as e:  # 네트워크 차단, 잘못된 모델 이름 등
        raise EmbeddingUnavailable(f"임베딩 모델 '{model}'을 불러오지 못함: {e}") from e

    def enc(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=float)

    enc.model_name = model  # type: ignore[attr-defined]
    return enc


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


class EmbedIndex:
    """라이브러리 사건 문장을 미리 벡터로 바꿔 두고, 새 기사와의 코사인 유사도(0 미만은 0)를 돌려준다."""

    def __init__(self, events: list[dict], encoder: Encoder):
        self.ids = [e["id"] for e in events]
        self.encoder = encoder
        self.model = getattr(encoder, "model_name", "custom")
        texts = [event_text(e) for e in events]
        self.mat = _normalize(np.asarray(encoder(texts), dtype=float)) if texts else np.zeros((0, 1))

    def query(self, text: str) -> dict[str, float]:
        if not self.ids:
            return {}
        q = _normalize(np.asarray(self.encoder([text]), dtype=float))[0]
        sims = np.clip(self.mat @ q, 0.0, 1.0)
        return {i: float(s) for i, s in zip(self.ids, sims)}


def blend(lexical: dict[str, float], embed: dict[str, float], weight: float) -> dict[str, float]:
    """(1 - w) × 단어 + w × 임베딩. 한쪽에만 있는 id는 없는 쪽을 0으로 본다. w=0이면 단어 유사도 그대로."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"embed_weight는 0~1이어야 합니다 (embed_weight={weight})")
    if weight == 0.0:
        return dict(lexical)
    keys = set(lexical) | set(embed)
    return {k: (1.0 - weight) * lexical.get(k, 0.0) + weight * embed.get(k, 0.0) for k in keys}
