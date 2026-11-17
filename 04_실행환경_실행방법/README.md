# 04. 실행환경·실행방법

검증 대상은 디노테퓨란 CatBoost 95열 · 검증 Optuna trial 3 · Window 90일 버전입니다. 공개 파일 무결성과 집계표는 Python 표준 라이브러리로 점검할 수 있습니다. 합성 예제 점검이나 CSV 예측을 실행하려면 아래 환경과 동결 모델 파일을 준비합니다.

## 1. 실행환경

재현에 사용하는 버전은 Python 3.11, CatBoost 1.2.10, NumPy 2.3.5, pandas 2.3.3입니다. 저장소 루트에서 [환경 정의](environment.yml)로 Conda 환경을 만들고 활성화합니다.

```powershell
conda env create -f "04_실행환경_실행방법/environment.yml"
conda activate dinotefuran-validation
```

## 2. 공개 패키지 점검

저장소 루트에서 실행합니다.

```powershell
python -B "01_소스코드/verify_package.py"
```

Windows PowerShell에서는 루트의 실행 스크립트로 같은 점검을 수행할 수 있습니다.

```powershell
.\검증_실행.ps1
```

점검 결과에서 공개 파일의 바이트와 SHA-256, 요약 집계 12행과 H별 집계 240행, 합성 예제의 입력 95열과 결과 3행을 확인합니다. 점검 결과를 파일로 남기려면 Python 실행에서 `--report-dir`, PowerShell에서 `-ReportDir`를 지정합니다. 예를 들어 `점검/예시`를 지정하면 그 안에 `검증_결과.json`을 저장합니다.

## 3. 동결 모델로 합성 예제 확인

승인 경로에서 받은 `model.cbm`을 로컬에 준비합니다. 아래 명령에서 예시 모델 경로를 실제 위치로 바꿉니다. 실행 환경과 파일 일치 기준은 [학습모델파일](../02_학습모델파일/README.md)에서 확인할 수 있습니다.

```powershell
python -B "01_소스코드/verify_package.py" --model "path/to/model.cbm" --report-dir "점검/예시"
.\검증_실행.ps1 -Model "path/to/model.cbm" -ReportDir "점검/예시"
```

두 명령은 같은 점검을 수행하므로 사용하는 셸에 맞는 명령 하나를 선택합니다. PowerShell에서 특정 Python을 사용하려면 `-PythonExecutable`에 실행 파일을 지정합니다. 점검 코드는 입력 3행을 동결 모델에 전달하고, [기대 출력](../06_입출력정의_샘플/example_output.csv)과 확률·임계값·판정을 대조합니다. 확률은 최대 절대차가 `1e-6` 이내인지 확인합니다.

## 4. 조립한 CSV로 한 번에 예측

저장소 루트에서 소스 폴더로 이동합니다. 다음 예시는 루트의 `local/model.cbm`을 사용해 합성 입력을 예측하고 루트에 `predictions.csv`를 저장합니다.

```powershell
cd "01_소스코드"
python -m final_model95 predict --model "../local/model.cbm" --input "../06_입출력정의_샘플/prepared_features_example.csv" --output "../predictions.csv"
```

다른 입력을 예측할 때는 `--model`, `--input`, `--output`에 실제 파일 경로를 지정합니다. 출력을 새 폴더에 저장할 경우 해당 폴더를 준비합니다. 스크립트는 준비된 95열로 추론하고 입력의 각 행에 확률·임계값·예측값을 붙입니다. 입력 조립에 필요한 열과 예측값의 의미는 [입출력 정의](../06_입출력정의_샘플/README.md)에서 확인할 수 있습니다.
