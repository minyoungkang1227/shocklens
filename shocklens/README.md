# ShockLens

지경학 뉴스 1건을 넣으면, 국부펀드형 글로벌 포트폴리오(12개 자산 바스켓)의 손실 위험(ES·VaR)이 어떻게 바뀌는지 근거와 함께 알려주는 멀티 에이전트입니다.
「국부펀드 KIC, AI 에이전트 공모전」(2026) 출품작입니다.

**설계 원칙: 숫자는 코드가, 판단은 LLM이.** 모든 수치는 결정론적 코드가 계산하고 계산 로그를 남기며, LLM은 뉴스 해석·계획·서술만 맡습니다.

## 현재 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. fetch | ETF 12개 가격(yfinance), 팩터 6개(FRED) 수집 | 완료 |
| 2. returns | 바스켓 수익률, 팩터 변동 계산 | 완료 |
| 3. betas | 베타 행렬 12×7 (평상시 / 위기 구간), 릿지 회귀 | 완료 |
| 4. library | 지경학 사건 라이브러리 15건 | 예정 |
| 5. quantiles | 강도→충격 크기 분위수 표 | 예정 |
| engine | 충격 계산, 블록 부트스트랩 ES/VaR | 예정 |
| agents | 헤드쿼터 + 에이전트 7개 (LangGraph) | 예정 (본선) |
| app | Streamlit 데모 화면 | 예정 (본선) |

## 실행 방법

### 1) 준비 (처음 한 번)

Python 3.11 이상과 Git이 필요합니다.

```bash
git clone https://github.com/minyoungkang1227/shocklens.git
cd shocklens

# 가상환경 만들기
python -m venv .venv
# 윈도우
.venv\Scripts\activate
# 맥
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) 테스트 (인터넷 없이 계산 로직 확인)

```bash
python -m pytest -q
```

### 3) 데이터 파이프라인 실행

```bash
python -m pipeline.run_all
```

끝나면 `data/model/` 에 `beta_normal.csv`, `beta_crisis.csv` 가 생깁니다. 데이터 파일은 Git에 올라가지 않으며 언제든 다시 만들 수 있습니다.

## 폴더 구조

```
config/      포트폴리오 비중, 팩터 정의, 임계값, 노출 매핑표 (YAML)
pipeline/    오프라인 데이터 구축 (수집 → 수익률 → 베타 → 사건 라이브러리 → 분위수)
engine/      순수 계산 (충격, 리스크, 계산 로그) — LLM 없음
agents/      헤드쿼터와 에이전트 7개
tools/       뉴스 검색, 본문 추출, 임베딩
app/         Streamlit 데모
tests/       단위 테스트
data/        파이프라인 산출물 (Git 제외)
```

## 데이터 출처

- ETF 가격: Yahoo Finance (yfinance 라이브러리, Apache-2.0). 데이터는 Yahoo 이용약관에 따라 개인·연구 용도로 사용
- 팩터: FRED, Federal Reserve Bank of St. Louis (VIXCLS, DTWEXBGS, DEXCHUS, DCOILWTICO, DGS10, BAMLH0A0HYM2)
- 뉴스 지수: Caldara & Iacoviello GPR 지수, Caldara et al. TPU 지수 (matteoiacoviello.com)

## 보안

API 키는 `.env` 파일에만 둡니다. `.env` 는 `.gitignore` 에 등록되어 있어 커밋되지 않습니다. `.env.example` 을 복사해서 사용하세요.

## 생성형 AI 활용

설계 논의와 코드 작성에 Claude(Anthropic)를 활용했습니다. 설계 판단, 결과 검증, 최종 수정은 작성자가 직접 수행합니다.
