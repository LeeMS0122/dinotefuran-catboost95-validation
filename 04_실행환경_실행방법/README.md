# 04. 실행환경·실행방법

검증 대상은 디노테퓨란 CatBoost 95열 · 검증 Optuna trial 3 · Window 90일 버전입니다. 아래 명령은 저장소를 내려받아 압축을 풀거나 복제한 뒤, `검증_실행.ps1`이 있는 루트 폴더에서 시작합니다.

| 작업 | 준비할 자료 | 실행 결과 |
| --- | --- | --- |
| 공개 패키지 점검 | 저장소 파일, Python 3.11 | 파일 크기·해시, 성능 집계와 합성 예제의 일관성을 점검. 콘솔과 선택한 JSON에 결과 표시 |
| 동결 모델로 예제 재현 | 공개 패키지, 지정한 CatBoost 환경, 해시가 일치하는 `model.cbm` | 합성 3행을 다시 추론하고 저장된 확률·판정과 대조. 선택한 JSON에 최대 확률 오차 기록 |
| 새 CSV로 예측 | 지정한 CatBoost 환경, 동결 모델, 95개 피처를 조립한 입력 CSV | 사용자가 지정한 경로에 원래 입력 열과 예측 3열을 포함한 CSV 저장 |

다운로드 직후에도 [합성 예제의 저장 결과](../06_입출력정의_샘플/example_output.csv)를 열어 볼 수 있습니다. 이 CSV는 동결 모델로 미리 계산해 배포한 파일입니다.

## 1. 실행환경

공개 패키지 점검은 Python 표준 라이브러리로 실행합니다. 모델 추론을 재현하는 환경은 Python 3.11, CatBoost 1.2.10, NumPy 2.3.5, pandas 2.3.3입니다. 루트에서 [환경 정의](environment.yml)로 Conda 환경을 만들고 활성화합니다.

```powershell
conda env create -f "04_실행환경_실행방법/environment.yml"
conda activate dinotefuran-validation
python --version
python -c "import catboost,numpy,pandas; print(catboost.__version__, numpy.__version__, pandas.__version__)"
```

## 2. 공개 패키지 점검

Windows PowerShell에서 다음과 같이 실행합니다. 기본 실행은 콘솔에 점검 결과를 표시합니다.

```powershell
.\검증_실행.ps1
```

현재 배포본에서 모델 경로 없이 점검을 마치면 아래와 같이 표시됩니다. 마지막 줄은 공개 자료 점검을 완료했고, 동결 모델로 확률을 새로 계산하는 단계는 생략됐음을 뜻합니다.

```text
Manifest: 31 files verified.
Aggregates: 12 summary rows, 240 H rows.
Synthetic contract: 95 model fields, 3 examples.
Native synthetic prediction: skipped (no --model supplied).
```

Python 명령으로도 같은 점검을 할 수 있습니다. 실행 결과 파일이 필요하면 PowerShell에서는 `-ReportDir`, Python에서는 `--report-dir`를 붙여 각각 실행합니다. 아래 두 줄은 서로 대체할 수 있는 명령입니다.

```powershell
.\검증_실행.ps1 -ReportDir "점검/공개자료"
python -B "01_소스코드/verify_package.py" --report-dir "점검/공개자료"
```

`점검/공개자료/검증_결과.json`은 기존 폴더가 없어도 생성됩니다. 정상 종료 시 `passed: true`와 함께 `manifest_files`, `summary_rows`, `horizon_rows`, `feature_count`, `synthetic_rows`로 각 점검의 수를 기록합니다. 공개 점검의 `native_model.status`는 `skipped`입니다. 특정 Python을 사용하려면 PowerShell 명령에 `-PythonExecutable "C:/경로/python.exe"`를 지정합니다.

공개 패키지 점검으로 확인하는 것은 배포 파일의 무결성, 공개 집계표의 계산 일관성, 합성 예제에 저장된 판정의 정확성입니다. 합성 3행의 확률을 다시 계산하려면 다음 절을 사용합니다. [파일 목록과 SHA-256](../09_검증대상버전_파일목록/README.md)

## 3. 동결 모델로 합성 예제 확인

승인된 경로에서 받은 `model.cbm`을 로컬에 준비합니다. 다음 예시는 `local/model.cbm`에 모델이 있을 때의 실행 방법입니다. 모델의 실제 위치에 맞춰 경로를 바꾸고, 활성화한 환경에서 명령 하나를 실행합니다.

```powershell
.\검증_실행.ps1 -Model "local/model.cbm" -ReportDir "점검/모델재현"
python -B "01_소스코드/verify_package.py" --model "local/model.cbm" --report-dir "점검/모델재현"
```

1. 공개 자료 점검을 완료한 뒤 모델 파일 해시를 설정의 `model_sha256`과 대조합니다. 모델을 불러온 후 피처 이름과 순서, 범주형 위치, 트리 수 및 클래스 순서도 확인합니다.
2. 합성 3행으로 새 확률을 계산하고, 임시 출력과 [기대 출력](../06_입출력정의_샘플/example_output.csv)의 원본 입력값, 행 순서, 확률·임계값·판정을 비교합니다. 각 행의 검출·불검출 판정은 같아야 하며, 확률의 최대 절대차 허용 범위는 `1e-6`입니다.
3. 일치하면 콘솔에 최대 절대차가 표시됩니다. 위 예시의 JSON 파일에는 `native_model.status: "passed"`와 함께 모델 SHA-256, 허용 오차 `probability_tolerance`, 실측 오차 `max_abs_error`가 기록됩니다.

검증 때 생성한 재예측 CSV는 점검 중에만 임시 폴더에 있습니다. 새 예측 CSV를 바탕으로 행별 결과를 따로 보관하려면 다음의 `predict` 명령을 사용합니다. 동결 모델의 식별 정보는 [모델 설정과 학습모델 안내](../02_학습모델파일/README.md)에 있습니다.

## 4. 조립한 CSV로 한 번에 예측

저장소 루트에서 소스 폴더로 이동합니다. 다음 예시는 루트의 `local/model.cbm`을 사용해 합성 입력을 예측하고 루트에 `predictions.csv`를 저장합니다.

```powershell
cd "01_소스코드"
python -m final_model95 predict --model "../local/model.cbm" --input "../06_입출력정의_샘플/prepared_features_example.csv" --output "../predictions.csv"
```

합성 예제를 사용하면 콘솔의 `Saved 3 predictions` 메시지에서 출력 경로를 확인할 수 있습니다. 루트의 `predictions.csv`를 열면 입력 3행 오른쪽에 `probability`, `threshold`, `prediction`이 추가되어 있습니다. 합성 예제의 경우 세 행 모두 확률 `0.13407738506793976`, 임계값 `0.0583612434566021`, 예측값 `1`을 기대합니다.

다른 입력을 예측할 때는 `--model`, `--input`, `--output`에 실제 경로를 지정합니다. 출력의 상위 폴더를 준비하고, 입력 CSV와 별도 경로를 사용합니다. 점검 스크립트의 `-ReportDir`는 검증 JSON의 위치이고, `predict`의 `--output`은 행별 예측 CSV의 위치입니다. [입력 95열과 결과 열의 정의](../06_입출력정의_샘플/README.md)

## 5. 점검이 실패했을 때 확인할 항목

| 증상 | 확인할 항목 |
| --- | --- |
| `python` 명령을 찾을 수 없음 | PowerShell이 사용하는 Python을 확인하고 환경을 활성화하거나 `-PythonExecutable`을 지정 |
| 파일 크기 또는 SHA-256 불일치 | 같은 배포본에서 내려받은 파일을 사용했는지, 편집기에서 인코딩이나 줄바꿈을 바꾸었는지 확인 |
| 모델 파일을 찾을 수 없거나 모델 SHA-256 불일치 | `-Model` 또는 `--model` 경로와 [동결 모델의 식별 정보](../02_학습모델파일/README.md)를 대조 |
| `catboost`, `numpy`, `pandas` 모듈을 찾을 수 없음 | Conda 환경을 활성화한 상태에서 모델 예제 재현 또는 예측 명령을 실행 |
| 필수 열 누락, 중복 헤더, 수치 또는 범주 코드 오류 | [입력 정의](../06_입출력정의_샘플/README.md)와 오류 메시지의 열 이름 또는 CSV 레코드 범위를 대조 |

검증 스크립트는 성공 시 종료 코드 0, 검사 실패 또는 요청한 보고서 저장 실패 시 1을 반환합니다. 실패 결과를 JSON으로 쓸 수 있는 경우 `passed: false`와 `error`를 확인합니다. 모델이 제공됐고 모델 추론·대조 단계에서 실패했다면 `native_model.status`는 `failed`입니다.
