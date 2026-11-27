"""학습형 1차 분류기 — Haiku 판정을 정답으로 삼아 작은 분류기를 가르친다(증류). 2026-11-27.

왜
  1차 분류(triage)는 LLM(Haiku)이 없으면 키워드 규칙(rule_triage)으로 떨어진다. 규칙은 사전에 없는 표현
  ("핵심 소재 반출 제한")을 놓치고, 데일리 워치는 처음부터 규칙만 쓴다. Haiku를 매번 부르면 대량 수집 때 비용이 든다.
  그래서 Haiku가 이미 판정해 둔 기사(library_builder collect의 articles.jsonl)로 작은 분류기를 학습해
  'LLM -> 학습 분류기 -> 규칙' 순서로 쓴다.

무엇을 학습하나 (기준선)
  다국어 문장 임베딩(MiniLM, engine/embedsim.py와 같은 모델) + L2 로지스틱 회귀. 라벨은 keep(통과/제외) 이진.
  CPU에서 몇 초면 학습되고, 라벨이 수백 건이어도 버틴다. LoRA(scripts/train_lora_triage.py)는 이 기준선을
  같은 시험지에서 확실히 이길 때만 붙인다.

나누기
  무작위가 아니라 기사 수집 월(seen) 기준. 과거 달은 학습, 최근 달은 시험. 같은 사건 기사가 양쪽에 갈라
  들어가 점수가 부풀려지는 누수를 막는다.

정답이 두 종류다
  - Haiku 라벨: 학습·시험 모두에 쓴다. 이걸로 잰 점수는 'Haiku를 얼마나 흉내 냈나'다.
  - 사람 라벨(data/eval/triage_gold.yaml): 시험셋에서 뽑은 100건을 사람이 확인한 것. 이걸로 잰 점수가
    '실제로 얼마나 맞나'다. Haiku 자신의 오답률도 여기서 처음 보인다.
  - 함정 시험지(data/eval/not_policy_cases.yaml): 정책처럼 보이는 시장 이벤트. 여기서 통과(keep) 비율은 낮을수록 좋다.

사용
  python -m pipeline.library_builder collect --start 2025-01-01 --end 2026-10-31   # (먼저) 기사 + Haiku 판정 쌓기
  python -m agents.triage_model status        # 데이터 몇 건, Haiku 판정 몇 건, 월별 분포
  python -m agents.triage_model gold          # 시험셋에서 사람 확인용 100건 뽑기 -> data/eval/triage_gold.yaml
  python -m agents.triage_model train         # 기준선 학습 -> data/model/triage_clf.npz
  python -m agents.triage_model eval [--lora data/model/lora_preds.jsonl]   # 규칙 / Haiku / 기준선 / LoRA 비교표
켜기: config/thresholds.yaml 의 triage.learned.enabled: true (eval 결과를 보고 사람이 켠다. 기본 꺼짐)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTICLES = ROOT / "data" / "library" / "builder" / "articles.jsonl"
GOLD = ROOT / "data" / "eval" / "triage_gold.yaml"
TRAPS = ROOT / "data" / "eval" / "not_policy_cases.yaml"
MODEL = ROOT / "data" / "model" / "triage_clf.npz"
DEFAULT_MIN_IMPORTANCE = 2


# ---------------------------------------------------------------- 데이터
def load_rows(path: Path = ARTICLES, llm_only: bool = True) -> list[dict]:
    """articles.jsonl을 읽는다. llm_only면 Haiku가 판정한 기사만(규칙 판정은 정답으로 쓰지 않는다)."""
    if not path.exists():
        return []
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r.get("text") and r.get("seen")]
    if llm_only:
        rows = [r for r in rows if (r.get("triage") or {}).get("mode") == "llm"]
    return rows


def month(r: dict) -> str:
    return str(r["seen"])[:7]


def split_by_month(rows: list[dict], test_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """수집 월 기준 시간 분할. 최근 달부터 거꾸로 test_frac 이상이 될 때까지 시험셋에 넣는다(최소 1개월).
    한 달은 통째로 한쪽에만 들어간다."""
    months = sorted({month(r) for r in rows})
    if len(months) < 2:
        raise ValueError(f"수집 월이 {len(months)}개뿐이라 시간 분할을 할 수 없습니다(2개월 이상 필요).")
    test_m, n = set(), 0
    for m in reversed(months[1:]):                 # 가장 오래된 달은 항상 학습
        test_m.add(m)
        n += sum(month(r) == m for r in rows)
        if n >= test_frac * len(rows):
            break
    return [r for r in rows if month(r) not in test_m], [r for r in rows if month(r) in test_m]


def label(r: dict) -> int:
    return int(bool(r["keep"]))


def load_gold(path: Path = GOLD) -> dict[str, int]:
    """사람 라벨 {id: 0/1}. gold가 비어 있는(null) 항목은 뺀다."""
    if not path.exists():
        return {}
    items = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("items") or []
    return {it["id"]: int(bool(it["gold"])) for it in items if it.get("gold") is not None}


def load_traps(path: Path = TRAPS) -> list[dict]:
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("cases") or []


# ---------------------------------------------------------------- 기준선: 임베딩 + 로지스틱 회귀 (numpy만)
def fit_logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 500, lr: float = 0.5) -> tuple[np.ndarray, float]:
    """L2 로지스틱 회귀(배치 경사하강). 클래스 불균형은 가중치로 맞춘다(통과 기사가 훨씬 적다)."""
    n, d = X.shape
    pos = max(y.sum(), 1)
    neg = max(n - y.sum(), 1)
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg))
    w, b = np.zeros(d), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * sw
        w -= lr * (X.T @ g / n + l2 * w / n)
        b -= lr * g.mean()
    return w, float(b)


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(X @ w + b)))


def _encoder(model_name: str | None = None, encoder=None):
    if encoder is not None:
        return encoder
    from engine import embedsim
    return embedsim.sentence_transformers_encoder(model_name or embedsim.DEFAULT_MODEL)


def _norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 0, n, 1.0)


def embed(texts: list[str], encoder) -> np.ndarray:
    from .injection import sanitize
    return _norm(np.asarray(encoder([sanitize(t)[:1000] for t in texts]), dtype=float))


def pick_threshold(p: np.ndarray, y: np.ndarray) -> float:
    """학습셋에서 F1이 가장 높은 문턱값(0.2~0.8). 시험셋은 보지 않는다."""
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.2, 0.81, 0.05):
        f = metrics(y, (p >= t).astype(int))["f1"]
        if f > best_f:
            best_t, best_f = float(t), f
    return round(best_t, 2)


def train(rows: list[dict] | None = None, encoder=None, out: Path = MODEL, test_frac: float = 0.2) -> dict:
    rows = load_rows() if rows is None else rows
    if not rows:
        raise SystemExit(f"학습 데이터가 없습니다: {ARTICLES}\n먼저 python -m pipeline.library_builder collect 를 Haiku 키가 있는 상태로 실행하세요.")
    tr, te = split_by_month(rows, test_frac)
    y = np.array([label(r) for r in tr])
    if y.sum() == 0 or y.sum() == len(y):
        raise SystemExit("학습셋 라벨이 한쪽뿐입니다(통과 0건 또는 제외 0건). 수집 기간이나 검색어를 넓히세요.")
    enc = _encoder(encoder=encoder)
    X = embed([r["text"] for r in tr], enc)
    w, b = fit_logreg(X, y)
    thr = pick_threshold(predict_proba(X, w, b), y)
    meta = {"model": getattr(enc, "model_name", "custom"), "threshold": thr, "n_train": len(tr), "n_test": len(te),
            "train_months": f"{month(min(tr, key=month))}~{month(max(tr, key=month))}",
            "test_months": f"{month(min(te, key=month))}~{month(max(te, key=month))}",
            "pos_rate": round(float(y.mean()), 3), "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, w=w, b=np.array([b]), meta=np.array([json.dumps(meta, ensure_ascii=False)]))
    return meta


# ---------------------------------------------------------------- 실사용: triage.classify가 부르는 경로
class LearnedTriage:
    """저장된 기준선 분류기. 인코더 로딩은 처음 한 번(무겁다)."""

    def __init__(self, path: Path = MODEL, encoder=None, threshold: float | None = None):
        z = np.load(path, allow_pickle=False)
        self.w, self.b = z["w"], float(z["b"][0])
        self.meta = json.loads(str(z["meta"][0]))
        self.threshold = float(threshold if threshold is not None else self.meta["threshold"])
        self.encoder = _encoder(self.meta.get("model"), encoder)

    def proba(self, texts: list[str]) -> np.ndarray:
        return predict_proba(embed(texts, self.encoder), self.w, self.b)

    def classify(self, text: str) -> dict:
        p = float(self.proba([text])[0])
        keep = p >= self.threshold
        # 이진 분류라 중요도는 두 값만 낸다: 통과면 3(전체 분석 대상), 아니면 1
        return {"relevant": keep, "importance": 3 if keep else 1, "p_keep": round(p, 3),
                "reason": f"학습 분류기 통과 확률 {p:.2f} (문턱 {self.threshold:.2f}, {self.meta.get('model')})"}


_CACHE: dict = {}


def get_learned(cfg: dict | None) -> tuple[LearnedTriage | None, str | None]:
    """설정이 켜져 있고 모델 파일이 있으면 분류기를, 아니면 (None, 이유)를 돌려준다. 실패도 캐시한다."""
    cfg = cfg or {}
    if not cfg.get("enabled"):
        return None, None
    path = ROOT / cfg.get("path", "data/model/triage_clf.npz")
    key = (str(path), cfg.get("threshold"))
    if key not in _CACHE:
        try:
            if not path.exists():
                raise FileNotFoundError(f"모델 파일 없음({path.name}). python -m agents.triage_model train")
            _CACHE[key] = (LearnedTriage(path, threshold=cfg.get("threshold")), None)
        except Exception as e:                       # 패키지 없음, 모델 다운로드 실패 등 -> 규칙으로
            _CACHE[key] = (None, f"{type(e).__name__}: {str(e)[:80]}")
    return _CACHE[key]


# ---------------------------------------------------------------- 평가
def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = np.asarray(y), np.asarray(yhat)
    tp = int(((y == 1) & (yhat == 1)).sum())
    fp = int(((y == 0) & (yhat == 1)).sum())
    fn = int(((y == 1) & (yhat == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"n": int(len(y)), "acc": float((y == yhat).mean()) if len(y) else 0.0, "precision": prec, "recall": rec, "f1": f1}


def rule_keep(text: str, min_importance: int = DEFAULT_MIN_IMPORTANCE) -> int:
    from .triage import passes, rule_triage
    return int(passes(rule_triage(text), min_importance))


def load_lora_preds(path: Path | None) -> dict[str, float]:
    """scripts/train_lora_triage.py가 쓴 예측 {id: p_keep}. 함정 사례는 id가 np_로 시작한다."""
    if not path or not Path(path).exists():
        return {}
    return {d["id"]: float(d["p_keep"]) for d in map(json.loads, Path(path).read_text(encoding="utf-8").splitlines()) if d}


def evaluate(rows: list[dict] | None = None, model: LearnedTriage | None = None, lora: dict[str, float] | None = None,
             gold: dict[str, int] | None = None, traps: list[dict] | None = None, test_frac: float = 0.2) -> dict:
    """같은 시험셋에서 규칙 / Haiku / 기준선 / LoRA를 비교한다. 반환: {'vs_haiku': {...}, 'vs_gold': {...}, 'traps': {...}}"""
    rows = load_rows() if rows is None else rows
    _, te = split_by_month(rows, test_frac)
    gold = load_gold() if gold is None else gold
    traps = load_traps() if traps is None else traps
    lora = lora or {}
    texts = [r["text"] for r in te]
    preds: dict[str, np.ndarray] = {
        "규칙": np.array([rule_keep(t) for t in texts]),
        "Haiku": np.array([label(r) for r in te]),
    }
    if model is not None:
        preds["기준선(임베딩+회귀)"] = (model.proba(texts) >= model.threshold).astype(int)
    if lora:
        preds["LoRA"] = np.array([int(lora.get(r["id"], 0.0) >= 0.5) for r in te])

    y_h = preds["Haiku"]
    out = {"vs_haiku": {k: metrics(y_h, v) for k, v in preds.items() if k != "Haiku"}}

    gi = [i for i, r in enumerate(te) if r["id"] in gold]
    if gi:
        y_g = np.array([gold[te[i]["id"]] for i in gi])
        out["vs_gold"] = {k: metrics(y_g, v[gi]) for k, v in preds.items()}

    if traps:
        tt = [c["text"] for c in traps]
        tr = {"규칙": np.array([rule_keep(t) for t in tt])}
        if model is not None:
            tr["기준선(임베딩+회귀)"] = (model.proba(tt) >= model.threshold).astype(int)
        if lora and all(c["id"] in lora for c in traps):
            tr["LoRA"] = np.array([int(lora[c["id"]] >= 0.5) for c in traps])
        out["traps"] = {k: {"n": len(tt), "통과(오답)": int(v.sum())} for k, v in tr.items()}
    out["split"] = {"n_test": len(te), "test_months": sorted({month(r) for r in te})}
    return out


def report_md(res: dict) -> str:
    L = [f"시험셋 {res['split']['n_test']}건 ({', '.join(res['split']['test_months'])})", ""]

    def table(title, d, note):
        L.extend([f"### {title}", note, "", "| 방법 | n | 정확도 | 정밀도 | 재현율 | F1 |", "|---|---|---|---|---|---|"])
        for k, m in d.items():
            L.append(f"| {k} | {m['n']} | {m['acc']:.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} |")
        L.append("")

    table("Haiku 판정 기준 (얼마나 흉내 냈나)", res["vs_haiku"], "정답 = Haiku의 keep. Haiku 자신은 1.00이라 뺐다.")
    if "vs_gold" in res:
        table("사람 라벨 기준 (실제로 얼마나 맞나)", res["vs_gold"], "정답 = data/eval/triage_gold.yaml. Haiku의 오답률도 여기서 보인다.")
    else:
        L.extend(["### 사람 라벨 기준", "아직 없음. `python -m agents.triage_model gold`로 뽑은 파일의 gold 칸을 채우면 생긴다.", ""])
    if "traps" in res:
        L.extend(["### 함정 시험지 (not_policy_cases.yaml)", "정책처럼 보이는 시장 이벤트. 통과 건수가 적을수록 좋다.", "",
                  "| 방법 | 사례 | 통과(오답) |", "|---|---|---|"])
        L.extend(f"| {k} | {v['n']} | {v['통과(오답)']} |" for k, v in res["traps"].items())
    return "\n".join(L)


def export_gold(rows: list[dict] | None = None, n: int = 100, path: Path = GOLD, seed: int = 0, test_frac: float = 0.2) -> int:
    """시험셋에서 n건(통과·제외 반반)을 뽑아 사람이 확인할 파일을 만든다. 이미 있으면 덮어쓰지 않는다."""
    if path.exists():
        raise SystemExit(f"{path} 가 이미 있습니다. 확인 중인 라벨을 지우지 않도록 덮어쓰지 않습니다.")
    rows = load_rows() if rows is None else rows
    _, te = split_by_month(rows, test_frac)
    rng = np.random.default_rng(seed)
    pos = [r for r in te if label(r)]
    neg = [r for r in te if not label(r)]
    k = min(n // 2, len(pos))
    pick = list(rng.permutation(pos)[:k]) + list(rng.permutation(neg)[:n - k])
    items = [{"id": r["id"], "seen": r["seen"], "title": r["title"], "haiku_keep": bool(r["keep"]),
              "haiku_reason": (r.get("triage") or {}).get("reason", ""), "gold": None} for r in pick]
    path.parent.mkdir(parents=True, exist_ok=True)
    head = ("# 1차 분류 사람 라벨 (시험셋에서 뽑음). gold 칸에 true(전체 분석 대상인 지경학 정책 사건) / false 를 적는다.\n"
            "# haiku_keep은 참고용이다. 그대로 베끼면 'Haiku를 흉내 낸 점수'와 '실제 점수'를 나눌 수 없다.\n")
    path.write_text(head + yaml.safe_dump({"items": items}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return len(items)


def status(rows_all: list[dict]) -> str:
    llm = [r for r in rows_all if (r.get("triage") or {}).get("mode") == "llm"]
    by_m: dict[str, list[int]] = {}
    for r in llm:
        by_m.setdefault(month(r), []).append(label(r))
    L = [f"기사 {len(rows_all)}건 (Haiku 판정 {len(llm)}건, 규칙 판정 {len(rows_all) - len(llm)}건 — 규칙 판정은 학습에 안 씀)"]
    L += [f"  {m}: {len(v)}건, 통과 {sum(v)}건" for m, v in sorted(by_m.items())]
    return "\n".join(L)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["status", "gold", "train", "eval"])
    ap.add_argument("--lora", help="LoRA 예측 jsonl (scripts/train_lora_triage.py 출력)")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    if a.cmd == "status":
        print(status(load_rows(llm_only=False)))
    elif a.cmd == "gold":
        print(f"{export_gold(n=a.n)}건 -> {GOLD}")
    elif a.cmd == "train":
        print(json.dumps(train(), ensure_ascii=False, indent=2))
    else:
        model = LearnedTriage() if MODEL.exists() else None
        if model is None:
            print("※ 기준선 모델이 없어 규칙·Haiku(·LoRA)만 비교합니다. python -m agents.triage_model train")
        print(report_md(evaluate(model=model, lora=load_lora_preds(Path(a.lora) if a.lora else None))))


if __name__ == "__main__":
    main()
