# Colab 실행 가이드

사용할 노트북은 `notebooks/run_experiment_lab_colab.ipynb`입니다.

## 1. GitHub 준비

압축파일의 최상위 내용 전체를 GitHub 저장소 루트에 올립니다. 노트북만 올리면
`src/`와 `config/`가 없으므로 실행되지 않습니다. 대회 데이터와 결과는 올리지
않습니다.

## 2. Drive 데이터 위치

```text
MyDrive/aimers_data/
├─ train.csv
├─ test.csv
└─ trackman_history.csv
```

## 3. 노트북 설정

```python
GITHUB_USER = "본인 GitHub 아이디"
REPO_NAME = "실제 저장소 이름"
BRANCH = "main"
REPO_IS_PRIVATE = True

MODE = "quick"
PRESET = "starter"
N_JOBS = 2
```

비공개 저장소는 복제 셀에서 Personal Access Token을 한 번 입력합니다. 토큰은
노트북 코드나 출력에 저장하지 않습니다.

## 4. 권장 실행 순서

1. `quick + starter`: 경로·전처리·실행 점검
2. `full + starter`: 피처 묶음 정식 비교
3. `full + extended`: 모델 및 제거 실험
4. 필요할 때만 `full + all`: XGBoost·CatBoost 추가

quick 결과는 시즌당 최대 5,000행을 쓰므로 모델 채택에 사용하지 않습니다.

## 5. 일부 실험만 실행

```python
ONLY_EXPERIMENTS = [
    "v1__lgbm_base",
    "v1_asof__lgbm_base",
]
```

동일 설정의 성공 실험은 자동으로 건너뜁니다. 강제로 다시 실행하려면
`RERUN=True`로 바꿉니다.

## 6. 결과

기본 저장 위치는 다음과 같습니다.

```text
MyDrive/aimers_results/experiment_lab/quick/
MyDrive/aimers_results/experiment_lab/full/
```

`leaderboard.csv`의 `brier_delta_vs_baseline`이 음수면 기존 V1 LightGBM보다
개선된 것입니다. 세부 예측·calibration·피처 중요도와 실패 로그는 `runs/`에
실험별로 저장됩니다.
