"""1차 분류 LoRA 학습 (선택, 2026-11-27) — 기준선(agents/triage_model.py)을 이길 때만 붙인다.

⚠ 이 스크립트는 작성 환경에서 실행해 보지 못했다(GPU·모델 다운로드 없음). 처음 돌릴 때 오류가 나면 그대로 알려줄 것.
권장 실행 환경: GPU가 있는 Colab. CPU에서도 돌지만 기사 수천 건이면 매우 느리다.

  pip install torch transformers peft
  python scripts/train_lora_triage.py                       # -> data/model/lora_preds.jsonl
  python -m agents.triage_model eval --lora data/model/lora_preds.jsonl

설정: Qwen2.5-0.5B(다국어) + 분류 헤드(AutoModelForSequenceClassification), LoRA r=8, alpha=16, task_type=SEQ_CLS.
분할은 triage_model.split_by_month와 똑같다(수집 월 기준, 최근 달이 시험). 시험셋과 함정 사례의 p_keep만 저장한다.
어댑터 가중치는 data/model/lora_triage/ 에 저장한다(데스크탑에서 실사용하려면 triage.classify에 연결하는 코드가 더 필요 — 기준선을 이겼을 때 작업).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import triage_model as tm  # noqa: E402
from agents.injection import sanitize  # noqa: E402


def main() -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default=str(ROOT / "data/model/lora_preds.jsonl"))
    a = ap.parse_args()

    rows = tm.load_rows()
    if not rows:
        raise SystemExit(f"학습 데이터 없음: {tm.ARTICLES}")
    tr, te = tm.split_by_month(rows)
    traps = tm.load_traps()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
    model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16, lora_dropout=0.1))
    model.print_trainable_parameters()
    model.to(dev)

    def batches(texts, labels=None, shuffle=False):
        idx = torch.randperm(len(texts)).tolist() if shuffle else list(range(len(texts)))
        for i in range(0, len(idx), a.bs):
            j = idx[i:i + a.bs]
            x = tok([sanitize(texts[k])[:2000] for k in j], return_tensors="pt", truncation=True,
                    max_length=a.max_len, padding=True).to(dev)
            yield x, (torch.tensor([labels[k] for k in j], device=dev) if labels is not None else None)

    texts, ys = [r["text"] for r in tr], [tm.label(r) for r in tr]
    pos = max(sum(ys), 1)
    weight = torch.tensor([len(ys) / (2 * (len(ys) - pos)), len(ys) / (2 * pos)], device=dev, dtype=torch.float)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    lossf = torch.nn.CrossEntropyLoss(weight=weight)            # 통과 기사가 적은 불균형 보정
    model.train()
    for ep in range(a.epochs):
        tot = 0.0
        for x, y in batches(texts, ys, shuffle=True):
            loss = lossf(model(**x).logits.float(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"epoch {ep + 1}: loss {tot:.3f}")

    model.eval()

    @torch.no_grad()
    def proba(tt):
        out = []
        for x, _ in batches(tt):
            out += torch.softmax(model(**x).logits.float(), -1)[:, 1].tolist()
        return out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, p in zip(te, proba([r["text"] for r in te])):
            f.write(json.dumps({"id": r["id"], "p_keep": round(p, 4)}) + "\n")
        for c, p in zip(traps, proba([c["text"] for c in traps])):
            f.write(json.dumps({"id": c["id"], "p_keep": round(p, 4)}) + "\n")
    model.save_pretrained(ROOT / "data/model/lora_triage")
    print(f"예측 {len(te)}건 + 함정 {len(traps)}건 -> {a.out}")


if __name__ == "__main__":
    main()
