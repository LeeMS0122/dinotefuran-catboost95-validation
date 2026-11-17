# 01. 소스코드

| 파일 | 역할 |
| --- | --- |
| [`final_model95/__main__.py`](final_model95/__main__.py) | 조립된 95열 입력 CSV의 열과 값을 점검하고, 동결 모델로 확률과 임계값 판정을 추가 |
| [`final_model95/verify_results.py`](final_model95/verify_results.py) | 94열·95열 공개 성능표의 분할, H 범위, 분모와 혼동행렬을 대조 |
| [`verify_package.py`](verify_package.py) | 배포 파일 SHA-256, 지표 집계, 합성 입출력을 한 번에 점검. 모델 파일을 지정하면 합성 입력을 다시 추론 |
| [`tests/`](tests/) | 합성 입력을 사용한 입력 스키마·판정·집계 검산 테스트 |

입력 조립 기준은 [03 데이터 분할과 입력 정의](../03_학습검증평가데이터_분할정보/README.md)와 [08 데이터 처리 및 누수점검](../08_데이터처리_누수점검/README.md)에 설명했습니다. Optuna 선정 조건과 trial별 결과는 [02 학습모델파일](../02_학습모델파일/README.md)에서 확인할 수 있습니다.

저장소 루트에서 공개 패키지를 점검합니다.

```bash
python -B "01_소스코드/verify_package.py"
```

[실행환경과 전체 명령 예시](../04_실행환경_실행방법/README.md)
