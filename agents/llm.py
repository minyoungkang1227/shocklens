"""LLM 연결부. 에이전트는 이 파일의 complete_json()만 부른다.

설정(.env):
  LLM_PROVIDER = anthropic | openai | none   (비우면 키가 있으면 anthropic, 없으면 none)
  LLM_API_KEY  = 발급받은 키
  LLM_MODEL    = 모델 이름 (비우면 공급자별 기본값)

provider가 none이면 LLM을 부르지 않고 None을 돌려준다.
그러면 각 에이전트는 규칙 기반 대체 경로로 동작한다. (키 없이도 전체 흐름을 시연·테스트할 수 있게)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4.1"}


def _load_dotenv(path: Path = Path(__file__).resolve().parents[1] / ".env") -> None:
    """python-dotenv 없이 .env를 읽는다. 이미 설정된 환경변수는 덮어쓰지 않는다."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def extract_json(text: str) -> dict:
    """모델 응답에서 JSON 객체만 꺼낸다. ```json 코드블록이나 앞뒤 설명이 붙어도 처리."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("응답에 JSON 객체가 없습니다")
    return json.loads(text[start:end + 1])


class LLM:
    def __init__(self, provider: str | None = None, model: str | None = None, api_key: str | None = None):
        _load_dotenv()
        key = api_key or os.environ.get("LLM_API_KEY", "")
        prov = (provider or os.environ.get("LLM_PROVIDER", "")).lower()
        if not prov:
            prov = "anthropic" if key else "none"
        if prov != "none" and not key:
            prov = "none"
        self.provider = prov
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL.get(prov, "")
        self._key = key
        self._client = None
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    def _client_obj(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._key)
            elif self.provider == "openai":
                import openai
                self._client = openai.OpenAI(api_key=self._key)
            else:
                raise ValueError(f"지원하지 않는 LLM_PROVIDER: {self.provider}")
        return self._client

    def complete_text(self, system: str, user: str, max_tokens: int = 2000) -> str | None:
        if not self.enabled:
            return None
        self.calls += 1
        c = self._client_obj()
        if self.provider == "anthropic":
            r = c.messages.create(model=self.model, max_tokens=max_tokens, temperature=0,
                                  system=system, messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        r = c.chat.completions.create(model=self.model, temperature=0, max_tokens=max_tokens,
                                      messages=[{"role": "system", "content": system},
                                                {"role": "user", "content": user}])
        return r.choices[0].message.content

    def complete_json(self, system: str, user: str, max_tokens: int = 2000, retries: int = 1) -> dict | None:
        """JSON 응답을 dict로 돌려준다. 파싱 실패하면 오류 내용을 붙여 한 번 더 요청한다."""
        if not self.enabled:
            return None
        msg = user
        for attempt in range(retries + 1):
            text = self.complete_text(system, msg, max_tokens)
            try:
                return extract_json(text or "")
            except (ValueError, json.JSONDecodeError) as e:
                if attempt == retries:
                    raise
                msg = f"{user}\n\n[이전 응답이 올바른 JSON이 아니었습니다: {e}] JSON 객체 하나만 출력하세요."
        return None
