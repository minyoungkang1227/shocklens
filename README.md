# ShockLens

지경학 뉴스 1건을 넣으면, 국부펀드형 글로벌 포트폴리오(12개 자산 바스켓)의 손실 위험(ES·VaR)이 어떻게 바뀌는지 근거와 함께 알려주는 멀티 에이전트입니다.
「국부펀드 KIC, AI 에이전트 공모전」(2026) 출품작입니다.

**설계 원칙: 숫자는 코드가, 판단은 LLM이.** 모든 수치는 결정론적 코드가 계산하고 계산 로그를 남기며, LLM은 뉴스 해석·계획·서술만 맡습니다.

## 현재 진행 상황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. fetch | ETF 13개 가격(yfinance), 팩터 5개(FRED) 수집 | 완료 |
| 2. returns | 바스켓 수익률, 팩터 변동 계산 | 완료 |
| 3. betas | 베타 행렬 12×7 (평상시 / 위기 구간), 릿지 회귀. 축소 강도는 시간 블록 교차검증으로 선택(`ridge_alpha: cv`, `python -m pipeline.betas --cv`로 확인) | 완료 |
| 4. library | 지경학 사건 라이브러리 26건(2002~2025). clean 13건 + 겹침이 거시 발표인 사건 9건은 가중치 0.5로 부분 사용, 전쟁·급락장 등 겹침 3건은 제외. 2012년 이전 사건은 팩터 반응만 | 진행 중 |
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

결과의 ES97.5는 세 숫자로 나눠 보여줍니다: ① 평상시(전체 기간 표본), ② 국면(위기면 VIX 상위 20% 표본), ③ ② + 이번 사건 충격. 그래서 '시장이 평소보다 위험한 정도(국면 효과)'와 '이번 사건이 더한 정도(사건 효과)'가 분리됩니다. 유사 사례는 세 층으로 찾습니다: 구조가 비슷한 사건(direct/relaxed) → 없으면 같은 수단 계열 전체(pool, 낮은 가중치) → 그것도 없으면 강도로 합성(synthetic). 반응 창에 FOMC·CPI 같은 거시 발표가 겹친 사건(`confounder_type: macro`)은 가중치 0.5로 쓰고, 전쟁·급락장이 겹친 사건(`other`)은 뺍니다(`--all-analogs`로 전체 사용). 유사 사건을 찾을 때 설명문은 발표 내용 문장만 쓰고 이후 전개는 뺍니다(결과를 알고 고르지 않기 위해).

충격 크기는 발표 직후 3거래일의 팩터 변동으로 잡습니다. 20거래일 누적은 같은 기간의 시장 전체 흐름이 섞여 사건 효과를 흐리기 때문입니다. 이후 20일의 변동성은 블록 부트스트랩 분포가 맡습니다.

선택한 사건을 새 뉴스로 가정하고, 나머지 사건을 유사 사례로 써서 스트레스테스트를 돌립니다(leave-one-out). 팩터 충격, 시나리오별 포트폴리오 수익률, 충격 전후 VaR95·ES97.5, 정규분포 대비 ES 과소평가, 손실 기여 상위 바스켓, 실제 사건 후 수익률을 출력하고 계산 장부(`data/calc_logs/`)에 기록합니다.

### 4-1) 백테스트 (예측 vs 실제)

```bash
python -m agents.backtest          # clean 사건을 하나씩 새 뉴스로 가정 -> 방향 적중률, 범위 적중률, 예측구간 커버·폭, severe 커버, 오차
python -m agents.backtest --no-asof  # 비교용: 전체 기간 베타·분위수 사용 (기본은 사건 이전 데이터로 재추정)
```

v1(구조 점수만), v2(문장 유사도·가중 분포·severe 하한선), v3(현재: + 겹친 사건 부분 사용·수단 풀)을 나란히 비교하고 `data/model/backtest.csv`에 저장합니다. 대상 사건보다 먼저 일어난 사건만 유사 사례로 쓰고, 베타와 분위수 표도 사건 이전 데이터만으로 다시 추정합니다(look-ahead 차단). '예측구간'은 3거래일 시나리오 범위에 사건 무관일 |3일 수익률|의 75% 분위(약 ±1%p)를 더한 것으로, 커버율은 폭과 같이 봐야 합니다.

```bash
python -m agents.relabel_severity   # 강도 6요소를 발표 당일 정보만으로 LLM이 다시 채점 (사후 편향 점검). thresholds severity.labels: blind 로 적용
```

### 4-2) 위약 검정 (사건 반응이 우연인가)

```bash
python -m engine.placebo           # 사건마다 3거래일 반응이 무작위 창 5,000개 중 몇 번째 백분위인지
```

백분위 80% 이상이면 평소 변동의 상위 20%보다 큰 반응입니다. 강도 점수와 실제 반응의 상관도 같이 출력합니다.

### 4-2b) 충격 크기 등급별 분석

```bash
python -m engine.size_report      # 소·중·대 등급별 과거 사건 반응 요약 (작은 사건끼리, 큰 사건끼리)
```

### 4-3) LLM 연결 점검

```bash
python -m agents.llm_check         # .env의 키로 연결 확인 + 샘플 뉴스 3건 구조화 결과
python -m agents.llm_check --full  # 전체 에이전트를 LLM 모드로 1회 실행
python -m agents.llm_compare       # LLM 있음/없음 비교: 사건 해석 정확도 + 백테스트 적중률 (결과 data/model/llm_compare.csv)
python -m agents.llm_compare --n 1 # 강도 자기일관성 끄고 빠르게 (호출 수 약 1/3)
python -m agents.analog_eval       # Analog Finder 코드 순위 vs Claude 재순위 (정답: data/library/analog_labels.yaml)
```

### 4-4) 사건 라이브러리 대량 구축 (GDELT + Claude Batch)

```bash
python -m pipeline.library_builder collect --start 2024-01-01 --end 2024-12-31   # 기사 수집 + 중요도 1차 분류
python -m pipeline.library_builder submit                                        # 라벨링 배치 제출 (비동기, 할인)
python -m pipeline.library_builder fetch                                         # 결과 -> data/library/candidates.yaml
python -m pipeline.library_builder label --limit 5                               # 배치 없이 몇 건만 바로 (시험)
```
후보의 clean 판정·날짜 확인은 사람이 하고, 검수한 사건만 events.yaml로 옮깁니다.

### 5) 에이전트 실행 (헤드쿼터 + 에이전트 7개)

```bash
python -m agents.run --event cn_ree_oct_2025 --trace            # 라이브러리 사건을 새 뉴스로 가정
python -m agents.run --file news.txt --trace                    # 뉴스 텍스트 파일 (PDF도 가능)
python -m agents.run --text "https://기사주소"                     # URL -> 본문 추출
python -m agents.run --text "China expands rare earth export controls"   # 헤드라인 -> 원문 기사 검색
python -m agents.run --text "..." --force                        # 중요도 1차 분류 무시
python -m agents.run --list-scenarios                            # 가상 시나리오 목록
python -m agents.run --scenario sc_taiwan_blockade --trace       # 가상 시나리오 (대만 해협 봉쇄 가정)
python -m agents.law_effect --event us_chips_2022                # 법령 장기 효과 (20/60/120거래일 업종 초과수익)
python -m agents.reverse --loss 0.05                             # 역스트레스테스트: 5% 손실을 내는 가장 그럴듯한 요인 충격
python -m agents.reverse --es 0.12 --factors tech_war,cny,risk_off --run   # ES 12% 목표, 지경학 채널만, 전체 실행
python -m agents.run --event cn_ree_oct_2025 --whatif "반도체 충격 3배"
python -m agents.run --event cn_ree_oct_2025 --ask "왜 반도체 손실이 제일 커?"   # 후속 질문 (계산 장부 조회)
python -m agents.run --event cn_ree_oct_2025 --chat             # 후속 질문 대화 모드
```

`.env`에 `LLM_API_KEY`가 있으면 Event Analyst(뉴스 해석)와 Reporter(브리핑 문장)가 LLM으로 동작하고, 없으면 키워드 규칙·고정 템플릿으로 동작합니다. 숫자는 어느 경우든 코드가 계산하고, Verifier가 브리핑의 모든 숫자·근거 태그를 계산 장부와 대조해 통과하지 못하면 담당 에이전트로 되돌립니다(최대 2회, 이후 "판단 보류"). 에이전트별 설계는 [docs/agents.md](docs/agents.md)에 있습니다.

### 5-1) 데일리 워치 (매일 국내·미국 핫토픽과 파급효과)

```bash
python -m agents.daily_watch --refresh          # 데이터 갱신 -> 국내·미국 RSS 수집 -> 토픽 선택 -> 정책 사건은 전체 분석
python -m agents.daily_watch --regions US --analyze 2
python -m agents.daily_watch --force            # 정책 사건이 아닌 토픽도 정량 분석 (참고용)
powershell -ExecutionPolicy Bypass -File scripts\register_daily_watch.ps1 -Time 07:40   # 매일 아침 자동 실행 등록
```

지역별 RSS 피드(`config/daily_watch.yaml`)에서 헤드라인을 모아 가장 많이 다뤄진 토픽 3개를 고릅니다. LLM이 있으면 토픽 묶기·정책 사건 판단·정성 파급 경로를 LLM이, 없으면 문장 유사도 규칙이 맡습니다. 정책 사건(지경학 + 국내 규제·선거)인 토픽은 ShockLens 전체 흐름으로 ES·사건 효과·3거래일 예측구간·손실 기여를 계산하고, 결과는 `data/daily/YYYY-MM-DD.md`(같은 날 다시 돌리면 `_v2`)와 `.json`에 저장됩니다.

### 5-2) 실전 성적표 (예측 기록 → 3거래일 뒤 자동 채점)

```bash
python -m agents.track_record                    # 반응 창이 끝난 예측을 채점하고 성적표 출력
python -m agents.track_record --add "뉴스" --news-date 2026-11-05   # 뉴스 하나를 예측으로 기록
python -m agents.track_record --list
```

데일리 워치가 분석한 토픽은 자동으로 `data/forecasts/forecasts.jsonl`에 기록되고(추가만, 수정 없음), 가격 데이터에 반응 창 3거래일이 들어오면 실제 포트폴리오 수익률로 채점됩니다(`scored.csv`). 기록 시점에 반응 창 첫날 장이 이미 열렸으면 '사후 해석'으로 따로 집계하고, 모델 실력은 '순수 예측'으로 봅니다. 데일리 워치 보고서 끝에 성적표가 붙습니다.

### 6) 데이터 사본 (선택)

```bash
python -m pipeline.snapshot        # data/processed, data/model -> data/snapshot (약 1MB)
```

Yahoo Finance 시세는 개인·연구 용도로만 쓰는 조건이라 수익률 사본(`data/snapshot`)은 저장소에 올리지 않습니다. 웹 앱은 사본이 없으면 처음 실행할 때 `pipeline.run_all`로 데이터를 직접 받습니다(2~3분). 본인 환경에서 빠르게 띄우려면 위 명령으로 사본을 만든 뒤 로컬에서만 쓰세요.

### 7) 데모 화면

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
tools/       news.py(뉴스 검색·기사 본문 추출), docs.py(PDF 읽기·핵심 문장 선별·언어 감지)
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
