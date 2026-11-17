# 아침 브리핑 2026-11-17 (화)

결론: 엔비디아 경영 공백 충격이 한미 증시를 같이 끌어내렸다. 코스피 -3.59%(5,100.50), 나스닥 -2.49%, VIX +19%. 새 정책 사건은 확인되지 않았다. 엔비디아 건은 Megaspeed 수출통제 문제에 걸려 있어 정책 사건으로 계속 추적한다.

## 국내 (11/16 마감)
1. 코스피 5,100.50(-190.03p, -3.59%). 외국인 1조3,844억원 순매도, 이 중 삼성전자 6,194억·SK하이닉스 2,242억(파이낸셜뉴스). SK하이닉스 -7.89%, 삼성전자 20만원 하회(재경일보). 코스닥 665.65(-4.86%)는 경제타임스 후속 기사 제목 기준, 본문 미확인. 시장 이벤트. 경로: SOXX·대만(반도체 상대수익), 외국인 이탈·위험회피에 따른 달러 강세.
2. '슈퍼위크' 대기: 엔비디아 실적(미국 17일)·앤스로픽 IPO 앞두고 외국인이 반도체 투톱을 팔고 SK스퀘어는 집중 매수(핀포인트뉴스, SBS). 시장 이벤트. 경로: 미국 주식 내 AI 대형주 수급, 상장 사모 밸류에이션 기준점.

## 미국 (11/16 마감)
1. 엔비디아 이사회가 젠슨 황 복귀를 검토하지 않는다는 CNBC 보도. NVDA -4.12%($156.14), SOXX -5.37%, 마이크론 -7.62%, SK하이닉스 ADR -7.29%. 다우 51,608.17(-1.41%), S&P500 7,416.59(-1.92%), 나스닥 24,509.76(-2.49%), VIX 23.05(+19.43%)(파이낸셜뉴스). 오늘 계기는 경영 이슈(시장 이벤트)지만 배경은 Megaspeed·Entity List(정책 사건, 수출통제). 경로: SOXX·대만·미국 주식, 위험회피.
2. 앤스로픽 IPO·엔비디아 실적·FOMC 의사록 대기로 AI 트레이드 전반 매도(Yahoo Finance, Reuters). 시장 이벤트. 경로: 미국 주식 AI 비중, 상장 사모.
3. 유가 3% 넘게 하락, 달러인덱스 2025년 4월 이후 최고, 미 국채 금리는 소폭 하락(Reuters 요약; 수치 모름). 시장 이벤트. 경로: 달러 강세는 비달러 바스켓(유럽·일본·중국·인도·해외 국채) 환산가치, 유가 하락은 인프라 일부, 금리 하락은 채권·부동산에 완충.

## 확인 메모
- Reuters 요약에 나온 "상무부의 AI칩 우회 조달 차단"은 2026-06-01 발표된 기존 조치로 확인. 새 정책 사건 아님.

## 정량 분석 대상
```
python -m agents.run --text "Nvidia board not considering Jensen Huang's return as CEO amid Megaspeed chip-diversion probe; chip stocks slide, KOSPI falls 3.6%" --trace
python -m agents.daily_watch --refresh
```

## Sources
- [파이낸셜뉴스 뉴욕증시 11/16](https://www.fnnews.com/news/202611170620543776)
- [파이낸셜뉴스 슈퍼위크 외국인 1.4조](https://www.fnnews.com/news/202611161610416501)
- [브릿지경제 마감시황](https://www.viva100.com/article/20261116500607)
- [재경일보](https://news.jkn.co.kr/post/1098988)
- [핀포인트뉴스](https://www.pinpointnews.co.kr/news/articleView.html?idxno=504644)
- [SBS](https://news.sbs.co.kr/news/endPage.do?news_id=N1008848523)
- [경제타임스](https://www.ket.kr/news/article.html?no=42096)
- [Yahoo Finance live 11/16](https://finance.yahoo.com/markets/live/stock-market-today-monday-november-16-dow-sp-500-nasdaq-nvidia-oil-091631084.html)
- [Reuters via MarketScreener](https://www.marketscreener.com/news/stocks-fall-ahead-of-nvidia-earnings-and-anthropic-ipo-oil-drops-ce785edfdd8cff25)
- [TNW 상무부 6/1 조치](https://thenextweb.com/news/us-nvidia-amd-china-subsidiary-chip-curbs)
