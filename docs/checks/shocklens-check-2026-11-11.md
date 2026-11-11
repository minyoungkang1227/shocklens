# ShockLens 점검 2026-11-11 — 데일리 워치 오분류 재현

## 증상
데일리 워치 국내 토픽 "젠슨 황 스캔들·미중 관계 냉각 우려"가 정책 사건으로 분류됨 → election_shift(US→US, AI 인프라)로 해석, 사건 효과 +1.53%p (calc_20261111_113846_96260f). 같은 토픽을 미국 섹션은 "정책 사건 아님"으로 판정.

## 재현 결과
| 입력 | triage | Event Analyst | 결과 |
|---|---|---|---|
| 헤드라인(영문), 강제 없음 | 1/5 [rule] → 생략 | - | skipped (맞음) |
| 연합뉴스 기사 URL, --force | 강제 | election_shift US→US, 근거 s7(중간선거·데이터센터 문장) | 통과, 사건 효과 +1.58%p, mild=central=severe=-1.93% (calc_20261111_120709_9f790b) |
| 헤드라인, --force | 강제 | domestic_regulation US→CN (3회 동일) | instrument 근거 없음 → 판단 보류 (calc_20261111_121309_5fa7c4) |

## 원인
1. Event Analyst에 "정책 사건 아님" 출구가 없음: 수단 enum에서 무조건 하나를 고름 → 기사 속 다른 정책 문장(s7)에 끌려가거나, 억지 수단(domestic_regulation US→CN)을 만듦
2. 데일리 워치에 토픽 제목 ↔ 사건 근거 문장 일치 검사 없음
3. 유사 사건 1건(event_level) 경로에서 severe 하한선이 작동하지 않아 범위가 한 점으로 붕괴
4. 근거 검사가 Verifier(Reporter 이후)에서만 일어나 재시도마다 Reporter 재실행(~30초·토큰), 같은 해석 3회 반복
5. 뉴스 검색: 도구 10회(설계 최대 5회), GDELT ReadTimeout으로 55~59초 대기
6. triage가 LLM 키가 있는데 [rule]로 판정 — Haiku 호출 실패 여부 확인 필요

## 수정 우선순위 (제안)
1. 수단 enum에 `none`(정책 사건 아님) 추가 → HQ가 "정책 사건 아님"으로 종료
2. 데일리 워치: 근거 문장과 토픽 제목의 문장 유사도가 기준 미만이면 분석 생략·경고
3. event_level 경로에도 severe 하한선 적용
4. 사건 게이트에서 근거 검사 선행, 재시도 해석이 직전과 같으면 즉시 판단 보류
5. 검색 도구 상한 5회, GDELT 타임아웃 단축 후 Google News RSS로 전환
6. triage Haiku 경로 로그 확인

## 참고 (11/11 데일리 워치)
- us_midterm_2026은 이미 라이브러리에 편입(가중 0.5), 성적표: 예측 중앙 -0.25% / 실제 -0.71% / 구간 적중, '사후 해석' 분류
- 국면: 최근 5거래일 VIX 평균 22.1 → 위기, 국면 효과 +5.69%p. 美 CPI 상회·10년물 5% 보도
- 연합뉴스 경제 피드 ConnectionError

## 반영 (fix30, 2026-11-11)
| # | 수정 | 위치 |
|---|---|---|
| 1 | 수단 enum에 `none` + `not_policy_reason`. 중심 사건이 정책 조치가 아니면 HQ가 `not_policy`로 종료(Reporter·재시도 없음). 규칙 경로도 정책 수단 단어가 없으면 같은 출구 | prompts, event_analyst.analyze, hq(n_not_policy) |
| 2 | 데일리 워치: 토픽 제목·헤드라인 vs 수단 근거 문장 유사도 < `topic_match_min`(0.1)이면 정량 결과 보류·성적표 기록 제외 | daily_watch.topic_match |
| 3 | 유사 사건 1~2건(event_level)이면 mild는 덜 나쁜 쪽, severe는 더 나쁜 쪽을 강도 분위수와 비교해 넓힘(`+widen`) | shock.widen_range, shock_mapper |
| 4 | LLM 해석의 필수 필드(수단·발동국·대상국)에 근거 문장이 없으면 Reporter 전에 재해석. 재해석이 직전과 같으면 바로 판단 보류 | hq.n_event |
| 5 | 헤드라인 검색 도구 호출 상한 5회(초과 시 도구가 거절), GDELT 타임아웃 4초 + 실패 시 10분간 건너뛰고 Google News RSS | news_input, tools/news |
| 6 | triage가 규칙으로 떨어질 때 원인(LLM 빈 응답 stop_reason·모델 / 예외 메시지)을 reason에 남김 | triage.classify |

테스트 130 → 137.

**재현 확인 (11/11 12:27, 데스크탑·LLM 경로):** 같은 연합뉴스 URL을 `--force`로 실행 → `not_policy` 종료(이유: 젠슨 황 CEO 거취 관련 기업 스캔들·주가 하락). Reporter·재시도 없음, Event Analyst 21초, 입력 14,412·출력 1,342 토큰. 이전에는 election_shift로 오분류되어 사건 효과 +1.58%p가 나왔음.

## 재검토 (11/12, 육칠·크래시 아웃 모드)
fix30의 주장을 하나씩 반례로 공격한 결과, 3건을 고침(fix31).
- **탈락: "규칙 경로에서 수단 단어가 없으면 정책 사건 아님"** — "정부가 반도체 기업 법인세 감면 확대 방안을 확정했다" 같은 실제 정책도 규칙 사전에 없으면 '정책 사건 아님'으로 단정됨. → 규칙 경로는 '못 찾음'으로 보고 판단 보류(재시도 없음). '정책 사건 아님' 판정은 LLM 경로만.
- **탈락(기존 버그 발견): 영어 "AI chips"가 산업 보조금(CHIPS법)으로 읽힘** — 정규식 `CHIPS`가 대소문자 무시로 일반명사 chips에 걸림. 미 상무부 칩 수출 제한 기사가 industrial_subsidy, 발동국 CN으로 해석됨. → `CHIPS`·`IRA`는 대문자만, 영어 수출통제 표현(restrict sales of chips, export curbs) 추가, Commerce Department·White House·Trump·Biden을 US로, 'to/against/on + 국가'를 대상국으로. textsim 같은 버그도 수정.
- **탈락: Streamlit 화면이 not_policy를 '추가 정보 필요'로 표시** → 생략과 같은 안내로 표시.
- 통과: LLM 경로 none 출구(11/11 데스크탑 재현), 근거 선행 검사(Verifier 기준과 동일), 재해석 반복 중단, event_level 범위 넓힘, 도구 상한, GDELT 우회, triage 원인 기록.
- 통과(측정): 토픽-근거 유사도 0.1 기준 — 실제 맞는 사례 4건 0.39~0.83, 오분류 사례 0.00.
- 회귀 확인: 라이브러리 37건 규칙 해석 정확도 수단 26/37·발동국 32/37로 수정 전과 같음. 테스트 137 → 138.
