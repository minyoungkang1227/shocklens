# 아침 브리핑 2026-11-15 (일)

결론: 주말이라 지난 24시간 동안 장이 열리지 않았다. 11/13(금) 장과 주말 전망 기사로 정리했다. 한미 공통 1번 토픽은 엔비디아 사태(젠슨 황 사임, Megaspeed를 통한 대중 칩 유출 의혹)다. 정책 사건은 두 개다. 미국 수출통제 논쟁, 그리고 미 중간선거 이후의 AI 데이터센터 규제 우려.

## 국내

1. 코스피가 장중 5,500을 넘었다가 5,290.53(-0.98%)로 마감했다. 외국인 7,034억원 순매도, 최대 순매도 종목은 SK하이닉스(-3.03%). 코스닥은 699.66(+0.45%)(파이낸셜뉴스). 시장 이벤트. 노출 바스켓: SOXX·대만(반도체 상대수익), 위험회피에 따른 달러.
2. 한국전력이 3분기 어닝쇼크로 9%대 급락했고, 전기·가스 업종은 -7.74%였다(아주경제). 시장 이벤트. 해외 포트폴리오로 가는 직접 경로는 약하다. 해외 인프라 바스켓 영향은 모름.
3. 미 중간선거 이후 AI 데이터센터 규제 우려가 남아 있다. 지난주 코스피는 +4.56%, 이번 주 증권가 밴드는 5,200~6,000. 정책 사건(election_shift). 노출 바스켓: 미국 주식 안의 AI 비중, SOXX, 인프라·부동산 중 데이터센터 자산(노출도 모름).

## 미국 (11/13 마감)

1. 엔비디아: 젠슨 황 사임. NVDA -2.01%(주간 -12.75%), AMD -10.95%. 나스닥 -0.91%, S&P500 -0.40%, 다우 +0.10%. Megaspeed(싱가포르, 2년간 48억달러 넘게 칩 매입)를 Entity List에 올리는 문제로 베선트와 러트닉이 대립하고, 상원은 12/3·12/10 청문회를 요구했다. 정책 사건(수출통제). 노출 바스켓: SOXX·대만·미국 주식. 미·중 회담으로 번지면 중국·위안화.
2. 10년물 5.15%(장중 5.17%, 10월 이후 최고). 10월 PPI 전년 대비 +6.5%. 시장은 12/8~9 FOMC에서 25bp 인상을 반영 중. 시장 이벤트. 노출 바스켓: 미 국채·해외 국채·회사채·인프라·부동산·상장 사모, 달러.
3. 앤스로픽 IPO 19일. 희망가 상단 기준 공모 규모 643.5억달러. 같은 주에 엔비디아 실적(17일)과 FOMC 의사록(한국시간 19일). 시장 이벤트. 노출 바스켓: 미국 주식 안 AI 대형주 수급, 상장 사모 밸류에이션 기준점.

## 정량 분석 대상

```
python -m agents.run --text "Nvidia CEO Jensen Huang resigns amid probe into AI chip diversion to China via Megaspeed; Treasury pushes Entity List listing" --trace
python -m agents.daily_watch --refresh
```

## 후속 처리 메모

- 11/12 브리핑에서 "중국 스파이 연루가 확인되면 정책 사건이 될 수 있음"으로 남겨 둔 엔비디아 건은 이번에 정책 사건(수출통제·Entity List)으로 바뀌었다. 사건 라이브러리 후보로 검토.

## Sources

- [파이낸셜뉴스 마감시황 11/13](https://www.fnnews.com/news/202611131625181132)
- [파이낸셜뉴스 주간증시전망](https://www.fnnews.com/news/202611150129458878)
- [한국경제](https://www.hankyung.com/article/2026111354636)
- [아주경제 한국전력](https://www.ajunews.com/view/20261113141552888)
- [이슈밸리](https://www.issuevalley.com/news/articleView.html?idxno=57825)
- [뉴스1](https://www.news1.kr/finance/market-exr/6344033)
- [TheStreet 11/13](https://www.thestreet.com/stock-market-today/stock-market-today-dow-jones-sp-500-nasdaq-updates-nov-13-2026)
- [Yahoo Finance live 11/13](https://finance.yahoo.com/markets/live/stock-market-today-friday-november-13-dow-sp-500-nasdaq-nvidia-ppi-141451558.html)
- [AI Weekly Megaspeed](https://aiweekly.co/alerts/bessent-confronts-lutnick-over-megaspeed-entity-list-delay)
- [Eurasia Business News](https://eurasiabusinessnews.com/2026/11/14/stock-market-today-november-13-2026-tech-stocks-fall-as-nvidia-earnings-and-anthropic-ipo-loom/)
