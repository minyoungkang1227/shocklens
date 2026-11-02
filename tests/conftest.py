import os

# 테스트는 네트워크를 쓰지 않는다: 캐시 길이 측정(count_tokens) 호출을 끈다. 캐싱 로직은 가짜 클라이언트로 따로 검사.
os.environ["LLM_CACHE"] = "0"
