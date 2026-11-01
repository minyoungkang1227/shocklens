# ShockLens

지경학 뉴스 1건을 넣으면, 국부펀드형 글로벌 포트폴리오(12개 자산 바스켓)의 손실 위험(ES·VaR)이 어떻게 바뀌는지 근거와 함께 알려주는 멀티 에이전트입니다.
「국부펀드 KIC, AI 에이전트 공모전」(2026) 출품작입니다.

**설계 원칙: 숫자는 코드가, 판단은 LLM이.** 모든 수치는 결정론적 코드가 계산하고 계산 로그를 남기며, LLM은 뉴스 해석·계획·서술만 맡습니다.

## 현재 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. fetch | ETF 13개 가격(yfinance), 팩터 5개(FRED) 수집 | 완료 |
| 2. returns | 바스켓 수익률, 팩터 변동 계산 | 완료 |
| 3. betas | 베타 행렬 12×7 (평상시 / 위기 구간), 릿지 회귀 | 완료 |
| 4. library | 지경학 사건 라이브러리 (13건, clean 8건, 목표 15건) | 진행 중 |
| 5. quantiles | 강도→충격 크기 분위수 표 | 완료 |
| engine | 충격 계산, 블록 부트스트랩 ES/VaR, 정규MC 비교, 계산 장부 | 완료 (실데이터 검증 전) |
| agents | 헤드쿼터 + 에이전트 7개 (LangGraph), 가드레일·검증·What-if | 1차 구현 (설계: docs/agents.md) |
| app | Streamlit 데모 화면 (`streamlit run app/streamlit_app.py`) | 1차 구현 |

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

끝나면 `data/model/` 에 베타 행렬, 사건 구간, 분위수 표가 생깁니다. 데이터 파일은 Git에 올라가지 않으며 언제든 다시 만들 수 있습니다.

### 4) 엔진 데모

```bash
python -m engine.demo --event cn_ree_2025            # 충격 크기: 발표 후 3거래일 반응 (기본)
python -m engine.demo --event cn_ree_2025 --window post  # 비교용: 20거래일 누적
```

결과의 ES97.5는 세 숫자로 나눠 보여줍니다: ① 평상시(전체 기간 표본), ② 국면(위기면 VIX 상위 20% 표본), ③ ② + 이번 사건 충격. 그래서 '시장이 평소보다 위험한 정도(국면 효과)'와 '이번 사건이 더한 정도(사건 효과)'가 분리됩니다. 유사 사례에는 반응 창에 다른 큰 충격이 겹치지 않은 사건(`clean: true`)만 씁니다(`--all-analogs`로 전체 사용).

충격 크기는 발표 직후 3거래일의 팩터 변동으로 잡습니다. 20거래일 누적은 같은 기간의 시장 전체 흐름이 섞여 사건 효과를 흐리기 때문입니다. 이후 20일의 변동성은 블록 부트스트랩 분포가 맡습니다.

선택한 사건을 새 뉴스로 가정하고, 나머지 사건을 유사 사례로 써서 스트레스테스트를 돌립니다(leave-one-out). 팩터 충격, 시나리오별 포트폴리오 수익률, 충격 전후 VaR95·ES97.5, 정규분포 대비 ES 과소평가, 손실 기여 상위 바스켓, 실제 사건 후 수익률을 출력하고 계산 장부(`data/calc_logs/`)에 기록합니다.

### 5) 에이전트 실행 (헤드쿼터 + 에이전트 7개)

```bash
python -m agents.run --event cn_ree_oct_2025 --trace            # 라이브러리 사건을 새 뉴스로 가정
python -m agents.run --file news.txt --trace                    # 뉴스 텍스트 파일
python -m agents.run --event cn_ree_oct_2025 --whatif "반도체 충격 3배"
```

`.env`에 `LLM_API_KEY`가 있으면 Event Analyst(뉴스 해석)와 Reporter(브리핑 문장)가 LLM으로 동작하고, 없으면 키워드 규칙·고정 템플릿으로 동작합니다. 숫자는 어느 경우든 코드가 계산하고, Verifier가 브리핑의 모든 숫자·근거 태그를 계산 장부와 대조해 통과하지 못하면 담당 에이전트로 되돌립니다(최대 2회, 이후 "판단 보류"). 에이전트별 설계는 [docs/agents.md](docs/agents.md)에 있습니다.

### 6) 데모 화면

```bash
streamlit run app/streamlit_app.py
```

브라우저가 열리면 과거 사건을 고르거나 뉴스를 붙여넣고 실행합니다. ES 3단 분해, 브리핑, 사건 카드, 유사 사건, 직접 노출, 팩터 충격, 에이전트 실행 기록, 검증 결과, What-if를 한 화면에서 봅니다. 데이터(`data/model`)가 없으면 처음 실행할 때 파이프라인을 자동으로 돌립니다.

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
- 팩터: FRED, Federal Reserve Bank of St. Louis (VIXCLS, DTWEXBGS, DEXCHUS, DCOILWTICO, DGS10). 신용 팩터는 ETF(HYG−IEF)로 대체: FRED의 하이일드 스프레드는 최근 3년치만 제공됨
- 뉴스 지수: Caldara & Iacoviello GPR 지수, Caldara et al. TPU 지수 (matteoiacoviello.com)

## 보안

API 키는 `.env` 파일에만 둡니다. `.env` 는 `.gitignore` 에 등록되어 있어 커밋되지 않습니다. `.env.example` 을 복사해서 사용하세요.

## 생성형 AI 활용

설계 논의와 코드 작성에 Claude(Anthropic)를 활용했습니다. 설계 판단, 결과 검증, 최종 수정은 작성자가 직접 수행합니다.
