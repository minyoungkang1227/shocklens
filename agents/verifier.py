"""Verifier: 브리핑을 계산 장부·근거와 대조해 통과/반려를 정한다.

검사 (전부 코드, 마지막 하나만 선택적으로 LLM)
1. 남은 플레이스홀더: {{...}} 가 채워지지 않았으면 -> Reporter
2. 숫자 대조: 본문의 모든 '%'·'%p' 숫자가 계산 결과(facts) 또는 뉴스 원문에 있는 값인지 -> Reporter
3. 태그: 숫자가 있는 문장은 [calc] 또는 [src] 필수, [src:id]는 실제 근거 목록에 있어야 함 -> Reporter
4. 무태그 문장: 소제목이 아닌 본문 문장에 태그가 하나도 없으면 -> Reporter
5. 방향 모순: 사건 효과가 +인데 "위험 감소"라고 쓰는 등 -> Reporter
5-2. 기간·가정 오표기: 3거래일 시나리오 수익률을 '20거래일'로, 정규분포 가정 ES를 '평상시'로 부르면 -> Reporter
5-3. 주입 의심 인용: 뉴스 안 지시문 의심 문장([src:s3] 등)을 근거로 인용하면 -> Reporter
6. 근거 부족: 사건의 필수 필드(수단·발동국·대상국)에 뉴스 근거가 없으면 -> Event Analyst
7. (선택) LLM 사실 검사: 근거 목록으로 뒷받침되지 않는 주장 문장 -> Reporter (warn 모드면 경고만)
   작성자와 검증자를 나누기 위해 Reporter와 다른 모델(기본 claude-haiku-5-5)을 쓴다.
"""
from __future__ import annotations

import json
import re

from .llm import LLM
from .state import Issue

NUM = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%p?")
TAG = re.compile(r"\[(?:src:[^\]]+|calc)\]")
SRC = re.compile(r"\[src:([^\]]+)\]")
UP_BAD = re.compile(r"(위험|ES|손실 위험|꼬리위험)[이가은는]?\s*(감소|줄어|낮아|완화)")
DOWN_BAD = re.compile(r"(위험|ES|손실 위험|꼬리위험)[이가은는]?\s*(증가|커지|높아|확대)")


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):                    # 목록 한 줄은 한 문장으로 본다
            out.append(line[2:])
            continue
        out += [s.strip() for s in re.split(r"(?<=\])\.?\s+(?=[^\s\[])|(?<=다\.)\s+(?!\[)", line) if s.strip()]
    return out


def check(final: str, facts: dict[str, str], sources: dict[str, str], risk_result: dict, event: dict,
          llm: LLM | None = None, use_llm: bool = False, llm_mode: str = "reject",
          require_structure: bool = False) -> dict:
    """llm: 작성자(Reporter)와 다른 모델을 넘기는 것을 권장 (설정 llm.verifier_model).
    llm_mode: reject = LLM 지적도 반려 / warn = 경고로만 기록하고 통과 여부엔 반영 안 함"""
    warnings: list[str] = []
    issues: list[Issue] = []
    # 1
    left = sorted(set(re.findall(r"\{\{([A-Z0-9_]+)\}\}", final)))
    if left:
        issues.append(Issue(kind="placeholder", detail=f"채워지지 않은 자리표시: {left}", route="reporter"))
    # 1-2 구성 누락: 본문(점검 목록 제외)에 필수 절과 계산 근거 문장이 충분한지 — 응답이 잘리면 숫자 검사만으로는 통과해 버린다
    body = final.split("### 점검 목록")[0]
    missing = [h for h in ("## 요약", "## 포트폴리오 영향") if h not in body]
    n_calc = sum(1 for s in _sentences(body) if "[calc]" in s)
    if require_structure and (missing or n_calc < 2):
        issues.append(Issue(kind="incomplete", detail=f"브리핑 구성 누락: 절 {missing or '있음'}, 계산 근거 문장 {n_calc}개(최소 2)",
                            route="reporter"))
    # 2
    allowed = set()
    for v in list(facts.values()) + list(sources.values()):     # 계산값 + 뉴스 원문에 있는 숫자만 허용
        for m in NUM.finditer(v):
            allowed.add(round(abs(float(m.group(1))), 2))
    for m in NUM.finditer(final):
        val = round(abs(float(m.group(1))), 2)
        if not any(abs(val - a) <= 0.011 for a in allowed):
            issues.append(Issue(kind="number_mismatch", detail=f"계산 장부에 없는 숫자: {m.group(0)}", route="reporter"))
    # 3, 4
    for s in _sentences(final):
        if NUM.search(s) and not TAG.search(s):
            issues.append(Issue(kind="untagged", detail=f"숫자 문장에 근거 태그 없음: {s[:50]}", route="reporter"))
        for sid in SRC.findall(s):
            for one in sid.split(","):
                if one.strip() not in sources:
                    issues.append(Issue(kind="unknown_source", detail=f"근거 목록에 없는 id: {one.strip()}",
                                        route="reporter"))
        if not TAG.search(s) and len(s) > 15 and "판단 보류" not in s:
            issues.append(Issue(kind="untagged", detail=f"근거 태그 없는 문장: {s[:50]}", route="reporter"))
    # 4-2 기간·가정 오표기: 3거래일 시나리오 값을 20거래일로, 정규분포 ES를 '평상시'로 부르는 경우
    other_vals = {v for k, v in facts.items() if not k.startswith("LOSS_")}
    loss_vals = {facts[k] for k in ("LOSS_MILD", "LOSS_CENTRAL", "LOSS_SEVERE") if facts.get(k)}
    # 다른 사실값과 글자가 겹치는 값(예: +0.00% 가 +0.00%p 안에)은 오탐을 막기 위해 '%p'가 붙지 않은 경우만 본다
    loss_rx = [re.compile(re.escape(v) + r"(?![p\d])") for v in loss_vals]
    nmc = facts.get("NORMAL_MC_ES")
    nmc_rx = re.compile(r"(?<![\d.])" + re.escape(nmc)) if nmc else None
    nmc_shared = bool(nmc) and any(v == nmc for k, v in facts.items() if k != "NORMAL_MC_ES")
    for s in _sentences(final):
        if any(r.search(s) for r in loss_rx) and "20거래일" in s and "3거래일" not in s:
            issues.append(Issue(kind="label", detail=f"3거래일 시나리오 수익률을 20거래일로 표기: {s[:50]}", route="reporter"))
        # (2026-11-12) 다른 사실값과 글자가 같으면(예: 정규 ES 2.00% = 중국 블록 비중 2.00%) 어느 값인지 알 수 없어 검사하지 않는다.
        # 앞에 숫자가 붙은 경우(12.00% 안의 2.00%)도 같은 값이 아니다.
        if nmc and nmc_rx.search(s) and "정규" not in s and not nmc_shared:
            issues.append(Issue(kind="label", detail=f"정규분포 가정 ES를 다른 이름으로 표기: {s[:50]}", route="reporter"))
    # 5-3 지시문 의심 문장 인용
    bad = {f["id"] for f in event.get("injection_flags") or []}
    for sid in SRC.findall(final):
        for one in sid.split(","):
            if one.strip() in bad:
                issues.append(Issue(kind="injection", detail=f"지시문 의심 문장 {one.strip()} 을 근거로 인용", route="reporter"))
    # 5
    eff = risk_result["decomposition"]["event_effect_ES"]
    if eff > 0.0005 and UP_BAD.search(final):
        issues.append(Issue(kind="direction", detail="사건 효과가 +인데 위험이 줄었다고 서술", route="reporter"))
    if eff < -0.0005 and DOWN_BAD.search(final):
        issues.append(Issue(kind="direction", detail="사건 효과가 -인데 위험이 커졌다고 서술", route="reporter"))
    # 6
    missing_ev = [f for f in ("instrument", "initiator", "target") if f in event.get("uncertain_fields", [])]
    if missing_ev:
        issues.append(Issue(kind="evidence", detail=f"뉴스 근거 없는 필수 필드: {missing_ev}", route="event_analyst"))
    # 7
    if use_llm and llm is not None and llm.enabled and not issues:
        try:
            res = llm.complete_json(
                "당신은 사실 검증관입니다. 브리핑 문장 중 주어진 근거(sources)로 뒷받침되지 않는 주장만 찾습니다. "
                "숫자와 [calc] 문장은 검사하지 않습니다. JSON {\"unsupported\": [문장 앞부분]} 만 출력.",
                json.dumps({"brief": final, "sources": sources}, ensure_ascii=False),
                schema={"type": "object", "properties": {"unsupported": {"type": "array", "items": {"type": "string"}}},
                        "required": ["unsupported"]}, name="emit")
            for s in (res or {}).get("unsupported", [])[:5]:
                if llm_mode == "warn":
                    warnings.append(f"[{getattr(llm, 'model', 'LLM')}] 근거로 뒷받침 안 될 수 있음: {s[:60]}")
                else:
                    issues.append(Issue(kind="unsupported", detail=f"근거로 뒷받침 안 됨: {s[:60]}", route="reporter"))
        except Exception as e:
            warnings.append(f"LLM 사실 검사 생략: {type(e).__name__}")
    # 중복 제거
    seen, uniq = set(), []
    for i in issues:
        key = (i.kind, i.detail)
        if key not in seen:
            seen.add(key)
            uniq.append(i)
    routes = [i.route for i in uniq]
    route = "event_analyst" if "event_analyst" in routes else ("shock_mapper" if "shock_mapper" in routes else
                                                             ("reporter" if routes else None))
    return {"passed": not uniq, "issues": [i.model_dump() for i in uniq], "route": route, "warnings": warnings,
            "checker_model": getattr(llm, "model", None) if use_llm and llm is not None and llm.enabled else None}
