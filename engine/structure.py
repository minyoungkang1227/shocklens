"""구조 점검: 블록별 비중(분절화), 국가 집중도, 국부펀드 고유 위험(관할국 노출). LLM 없음.

- 관할국 비중: 바스켓 비중 x 바스켓별 관할국 근사 비율 (config/jurisdictions.yaml)
- 블록 비중: 관할국을 미국 블록 / 중국 블록 / 비동맹·연결국으로 묶음
- 집중도: 국가 비중의 HHI와 유효 국가 수(1/HHI)
- 꼬리손실 블록 기여: ES 구간 경로의 바스켓 기여를 블록으로 합산
- 국부펀드 고유 위험: 사건 당사국 중 상대 블록(기본 중국 블록) 관할에 있는 자산 비중
  = 대응조치·자본통제·동결에 직접 노출될 수 있는 몫. 국부펀드·중앙은행 자산이 표적이면 선례 위험 표시
"""
from __future__ import annotations


def jurisdiction_shares(weights: dict[str, float], jmap: dict[str, dict[str, float]]) -> dict[str, float]:
    tot = sum(weights.values()) or 1.0
    out: dict[str, float] = {}
    for b, w in weights.items():
        for c, s in (jmap.get(b) or {"GLOBAL": 1.0}).items():
            out[c] = out.get(c, 0.0) + w / tot * s
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def to_blocs(shares: dict[str, float], blocs: dict[str, list[str]]) -> dict[str, float]:
    out = {k: 0.0 for k in blocs}
    for c, s in shares.items():
        k = next((b for b, cs in blocs.items() if c in cs), list(blocs)[-1])
        out[k] += s
    return out


def hhi(shares: dict[str, float]) -> float:
    return float(sum(s * s for s in shares.values()))


def tail_by_bloc(tail: dict[str, float], jmap: dict, blocs: dict) -> dict[str, float]:
    by_c: dict[str, float] = {}
    for b, c_share in tail.items():
        for c, s in (jmap.get(b) or {"GLOBAL": 1.0}).items():
            by_c[c] = by_c.get(c, 0.0) + c_share * s
    return to_blocs(by_c, blocs)


def structure(weights: dict[str, float], tail: dict[str, float], event: dict, cfg: dict) -> dict:
    jmap, blocs = cfg.get("baskets", {}), cfg.get("blocs", {})
    js = jurisdiction_shares(weights, jmap)
    bl = to_blocs(js, blocs) if blocs else {}
    h = hhi(js)
    tb = tail_by_bloc(tail, jmap, blocs) if blocs and tail else {}
    tgt = [c for c in event.get("target", []) if c != "GLOBAL"]
    init = event.get("initiator")
    rival = set(blocs.get(cfg.get("rival_bloc", "중국 블록"), []))
    adverse = [c for c in dict.fromkeys([init, *tgt]) if c in rival]      # 사건 당사국 중 상대 블록 관할
    out = {
        "country_shares": {c: round(s, 4) for c, s in js.items()},
        "bloc_shares": {k: round(v, 4) for k, v in bl.items()},
        "tail_by_bloc": {k: round(v, 4) for k, v in tb.items()},
        "hhi": round(h, 4), "effective_countries": round(1 / h, 1) if h > 0 else None,
        "target_share": round(sum(js.get(c, 0.0) for c in tgt), 4),
        "initiator_share": round(js.get(init, 0.0), 4),
        "flashpoint_share": round(sum(js.get(c, 0.0) for c in cfg.get("flashpoints", [])), 4),
        "adverse_countries": adverse,
        "adverse_share": round(sum(js.get(c, 0.0) for c in adverse), 4),
        "sovereign_flag": bool(event.get("sovereign_asset_risk")),
        "target_countries": tgt, "initiator": init,
    }
    return out
