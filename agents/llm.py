"""LLM 연결부. 에이전트는 이 파일의 complete_json() / complete_text() / chat_tools()만 부른다.

구조화 출력: schema를 주면 anthropic tool use로 JSON을 받는다 (tool_choice로 도구 호출 강제).
  모델이 input_schema(어휘는 enum)에 맞춘 인자만 내므로 정규식 파싱·재요청이 필요 없다.
  schema가 없거나 다른 공급자면 예전처럼 텍스트에서 JSON을 꺼낸다.

설정(.env):
  LLM_PROVIDER = anthropic | openai | none   (비우면 키가 있으면 anthropic, 없으면 none)
  LLM_API_KEY  = 발급받은 키
  LLM_MODEL    = 모델 이름 (비우면 공급자별 기본값)

provider가 none이면 LLM을 부르지 않고 None을 돌려준다.
그러면 각 에이전트는 규칙 기반 대체 경로로 동작한다. (키 없이도 전체 흐름을 시연·테스트할 수 있게)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4.1"}
SMALL_MODEL = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4.1-mini"}
# 프롬프트 캐시 최소 길이 (공식 문서 2026-11 기준: Sonnet 5.5·Haiku 5.5·Opus 5.5 = 512토큰, Sonnet 5 = 1,024)
CACHE_MIN_TOKENS = {"claude-sonnet-5-5": 512, "claude-haiku-5-5": 512, "claude-opus-5-5": 512, "claude-sonnet-5": 1024}
_TOKEN_CACHE: dict[str, int] = {}


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
        self.last_meta: dict = {}          # 마지막 호출의 종료 사유·글자 수 (실패 원인 추적용)
        self.cache = os.environ.get("LLM_CACHE", "1") != "0"     # 시스템 프롬프트 캐싱 (길이가 최소 기준 이상일 때만)
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def derive(self, model: str | None = None, small: bool = False) -> "LLM":
        """같은 키·공급자로 모델만 바꾼 LLM (예: Verifier·중요도 분류는 작은 모델)."""
        m = model or (SMALL_MODEL.get(self.provider) if small else None) or self.model
        child = LLM(provider=self.provider, model=m, api_key=self._key or "none")
        child.cache = self.cache
        return child

    def _track(self, r) -> None:
        u = getattr(r, "usage", None)
        if u is None:
            return
        for k, a in (("input", "input_tokens"), ("output", "output_tokens"),
                     ("cache_read", "cache_read_input_tokens"), ("cache_write", "cache_creation_input_tokens")):
            v = getattr(u, a, 0)
            self.usage[k] += v if isinstance(v, int) else 0

    def system_tokens(self, system: str) -> int:
        """시스템 프롬프트 토큰 수. count_tokens API로 한 번 재고 기억한다 (실패하면 글자 수로 보수적 추정)."""
        key = hashlib.md5(f"{self.model}|{system}".encode()).hexdigest()
        if key not in _TOKEN_CACHE:
            try:
                r = self._client_obj().messages.count_tokens(model=self.model, system=system,
                                                             messages=[{"role": "user", "content": "."}])
                _TOKEN_CACHE[key] = int(r.input_tokens)
            except Exception:
                _TOKEN_CACHE[key] = len(system) // 3          # 한국어 혼합 문서의 하한 추정
        return _TOKEN_CACHE[key]

    def system_param(self, system: str):
        """캐시 기준을 넘는 시스템 프롬프트면 cache_control을 붙인 블록으로, 아니면 문자열 그대로."""
        if not (self.cache and self.provider == "anthropic" and system):
            return system
        if self.system_tokens(system) < CACHE_MIN_TOKENS.get(self.model, 1024):
            return system
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

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

    def anthropic_kwargs(self, system: str, user: str, max_tokens: int) -> dict:
        return {"model": self.model, "max_tokens": max_tokens, "system": self.system_param(system),
                "messages": [{"role": "user", "content": user}]}

    def complete_text(self, system: str, user: str, max_tokens: int = 2000) -> str | None:
        if not self.enabled:
            return None
        self.calls += 1
        c = self._client_obj()
        if self.provider == "anthropic":
            # anthropic SDK 2.x는 temperature 인자를 받지 않는다. 결정성은 코드 가드레일·검증이 맡는다.
            r = c.messages.create(**self.anthropic_kwargs(system, user, max_tokens))
            self._track(r)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            self.last_meta = {"stop_reason": getattr(r, "stop_reason", None), "chars": len(text),
                              "blocks": [getattr(b, "type", "?") for b in r.content]}
            return text
        r = c.chat.completions.create(model=self.model, temperature=0, max_tokens=max_tokens,
                                      messages=[{"role": "system", "content": system},
                                                {"role": "user", "content": user}])
        return r.choices[0].message.content

    def complete_tool(self, system: str, user: str, schema: dict, name: str = "emit",
                      description: str = "결과를 구조화해 제출한다.", max_tokens: int = 2000) -> dict:
        """도구 호출을 강제해 schema에 맞는 dict를 받는다 (anthropic 전용)."""
        self.calls += 1
        c = self._client_obj()
        kw = self.anthropic_kwargs(system, user, max_tokens)
        kw["tools"] = [{"name": name, "description": description, "input_schema": schema}]
        # claude-sonnet-5-5 등 일부 모델은 tool_choice "tool"/"any"(호출 강제)를 받지 않는다 -> auto로 보내고
        # 프롬프트로 도구 사용을 지시한다. 모델이 도구 대신 본문으로 답하면 본문에서 JSON을 꺼낸다.
        kw["tool_choice"] = {"type": "auto"}
        kw["messages"][0]["content"] = user + f"\n\n반드시 {name} 도구를 호출해 결과를 제출하세요."
        r = c.messages.create(**kw)
        self._track(r)
        blocks = [b for b in r.content if getattr(b, "type", "") == "tool_use" and b.name == name]
        self.last_meta = {"stop_reason": getattr(r, "stop_reason", None), "mode": "tool" if blocks else "tool_text",
                          "blocks": [getattr(b, "type", "?") for b in r.content]}
        if blocks:
            return dict(blocks[0].input)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        try:
            return extract_json(text)
        except (ValueError, json.JSONDecodeError) as e:
            raise ValueError(f"도구 호출도 JSON도 없습니다 (stop_reason={self.last_meta['stop_reason']}): {e}")

    def complete_cited(self, system: str, user: str, chunks: list[str], title: str = "뉴스 원문",
                       max_tokens: int = 4000) -> list[dict] | None:
        """인용 기능: chunks(문장 목록)를 custom content 문서로 넣고, 응답을 [{text, cites:[chunk 번호]}]로 돌려준다.
        인용은 구조화 출력과 같이 쓸 수 없으므로 일반 텍스트 응답에만 쓴다."""
        if not self.enabled or self.provider != "anthropic" or not chunks:
            return None
        self.calls += 1
        doc = {"type": "document", "title": title, "citations": {"enabled": True},
               "source": {"type": "content", "content": [{"type": "text", "text": t} for t in chunks]}}
        r = self._client_obj().messages.create(
            model=self.model, max_tokens=max_tokens, system=self.system_param(system),
            messages=[{"role": "user", "content": [doc, {"type": "text", "text": user}]}])
        self._track(r)
        out = []
        for b in r.content:
            if getattr(b, "type", "") != "text":
                continue
            cites = []
            for c in getattr(b, "citations", None) or []:
                if getattr(c, "type", "") == "content_block_location":
                    cites += list(range(c.start_block_index, max(c.end_block_index, c.start_block_index + 1)))
            out.append({"text": b.text, "cites": sorted(set(cites))})
        self.last_meta = {"stop_reason": getattr(r, "stop_reason", None), "mode": "citations",
                          "chars": sum(len(o["text"]) for o in out)}
        return out

    def chat_tools(self, system: str, messages: list[dict], tools: list[dict], handler,
                   max_rounds: int = 4, max_tokens: int = 1500) -> tuple[str, list[dict]]:
        """도구를 쓰는 대화 한 턴. 모델이 도구를 부르면 handler(name, input)->dict 결과를 돌려주고 반복한다.
        반환: (최종 답변 텍스트, 호출 기록 [{name, input, output}])"""
        if not self.enabled or self.provider != "anthropic":
            return "", []
        c = self._client_obj()
        msgs = list(messages)
        log: list[dict] = []
        for _ in range(max_rounds):
            self.calls += 1
            r = c.messages.create(model=self.model, max_tokens=max_tokens, system=self.system_param(system),
                                  messages=msgs, tools=tools)
            self._track(r)
            uses = [b for b in r.content if getattr(b, "type", "") == "tool_use"]
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            if not uses:
                return text, log
            content = []                                     # 응답 블록을 API 입력 형식으로 되돌린다 (불필요한 필드 제외)
            for b in r.content:
                if getattr(b, "type", "") == "text" and b.text:
                    content.append({"type": "text", "text": b.text})
                elif getattr(b, "type", "") == "tool_use":
                    content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
            msgs.append({"role": "assistant", "content": content})
            results = []
            for u in uses:
                try:
                    out = handler(u.name, dict(u.input))
                except Exception as e:                       # 도구 오류도 모델에게 알려 스스로 고치게 한다
                    out = {"error": str(e)}
                log.append({"name": u.name, "input": dict(u.input), "output": out})
                results.append({"type": "tool_result", "tool_use_id": u.id,
                                "content": json.dumps(out, ensure_ascii=False, default=str)})
            msgs.append({"role": "user", "content": results})
        return "도구 호출 한도에 도달해 답변을 마치지 못했습니다 (판단 보류).", log

    def complete_json(self, system: str, user: str, max_tokens: int = 2000, retries: int = 1,
                      schema: dict | None = None, name: str = "emit") -> dict | None:
        """JSON 응답을 dict로 돌려준다.
        schema가 있고 anthropic이면 tool use로 받는다(파싱 실패 없음). 아니면 텍스트에서 꺼내고 실패 시 재요청."""
        if not self.enabled:
            return None
        if schema is not None and self.provider == "anthropic":
            return self.complete_tool(system, user, schema, name, max_tokens=max_tokens)
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
