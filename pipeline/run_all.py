"""파이프라인 전체 실행: python -m pipeline.run_all

1. fetch   : ETF 가격, FRED 팩터 수집
2. returns : 수익률·팩터 변동 계산
3. betas   : 베타 행렬(normal / crisis) 추정
(4. 사건 라이브러리, 5. 분위수 표는 다음 단계에서 추가)
"""
from . import betas, fetch, returns


def main() -> None:
    fetch.run()
    returns.run()
    betas.run()
    print("[done] data/model/ 에 베타 행렬이 저장됐습니다.")


if __name__ == "__main__":
    main()
