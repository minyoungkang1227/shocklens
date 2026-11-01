"""파이프라인 전체 실행: python -m pipeline.run_all

1. fetch     : ETF 가격, FRED 팩터 수집
2. returns   : 수익률·팩터 변동 계산
3. betas     : 베타 행렬(normal / crisis) 추정
4. library   : 사건별 반응 구간 계산 (data/library/events.yaml)
5. quantiles : 강도 -> 충격 크기 분위수 표
"""
from . import betas, fetch, library, quantiles, returns


def main() -> None:
    fetch.run()
    returns.run()
    betas.run()
    library.run()
    quantiles.run()
    print("[done] data/model/ 에 산출물이 저장됐습니다. 다음: python -m engine.demo")


if __name__ == "__main__":
    main()
