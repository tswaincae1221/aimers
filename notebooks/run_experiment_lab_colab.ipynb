{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LG Aimers 피처·모델 자동 실험실\n",
    "\n",
    "팀 공통 기준인 **2019~2023 학습 → 2024 검증**으로 여러 피처 묶음과 모델을 비교합니다. 각 실험의 점수, 기준 모델 대비 개선폭, 실행 시간, 예측값, calibration, 피처 중요도와 리더보드 그래프를 Google Drive에 자동 저장합니다.\n",
    "\n",
    "처음에는 반드시 `MODE='quick'`, `PRESET='starter'`로 코드만 점검하세요. quick 점수는 표본 점수라 모델 채택에 사용하지 않습니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pathlib import Path\n",
    "\n",
    "# ===== 사용자 설정 =====\n",
    "GITHUB_USER = \"tswaincae1221\"\n",
    "REPO_NAME = \"lg-aimers-experiment-lab\"  # 이 압축파일을 올린 실제 저장소 이름\n",
    "BRANCH = \"main\"\n",
    "REPO_IS_PRIVATE = True\n",
    "\n",
    "DRIVE_DATA_DIR = Path(\"/content/drive/MyDrive/aimers_data\")\n",
    "DRIVE_RESULT_DIR = Path(\"/content/drive/MyDrive/aimers_results/experiment_lab\")\n",
    "\n",
    "MODE = \"quick\"       # quick 점검 후 full로 변경\n",
    "PRESET = \"starter\"   # starter → extended → all 순서 권장\n",
    "ONLY_EXPERIMENTS = []   # 예: [\"v1__lgbm_base\", \"v1_asof__lgbm_base\"]\n",
    "N_JOBS = 2\n",
    "RERUN = False\n",
    "\n",
    "REPO_URL = f\"https://github.com/{GITHUB_USER}/{REPO_NAME}.git\"\n",
    "REPO_DIR = Path(\"/content/lg-aimers-experiment-lab\")\n",
    "LOCAL_DATA_DIR = REPO_DIR / \"data\"\n",
    "RESULT_DIR = DRIVE_RESULT_DIR / MODE\n",
    "print(\"저장소:\", REPO_URL)\n",
    "print(\"실행 모드:\", MODE, \"/ 프리셋:\", PRESET)\n",
    "print(\"결과 폴더:\", RESULT_DIR)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from google.colab import drive\n",
    "drive.mount(\"/content/drive\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 저장소 불러오기\n",
    "\n",
    "비공개 저장소라면 GitHub Personal Access Token을 입력합니다. 입력값은 화면에 표시되지 않습니다. 이미 복제된 폴더가 있으면 다시 복제하지 않습니다. 코드가 갱신됐는데 예전 폴더가 남아 있다면 Colab 런타임을 재시작한 뒤 다시 실행하세요."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import base64\n",
    "import getpass\n",
    "import subprocess\n",
    "\n",
    "if not GITHUB_USER or not REPO_NAME:\n",
    "    raise ValueError(\"GITHUB_USER와 REPO_NAME을 설정하세요.\")\n",
    "\n",
    "if (REPO_DIR / \"src/experiment_runner.py\").is_file():\n",
    "    print(\"기존 저장소 사용:\", REPO_DIR)\n",
    "else:\n",
    "    command = [\"git\", \"clone\", \"--branch\", BRANCH, \"--depth\", \"1\", REPO_URL, str(REPO_DIR)]\n",
    "    if REPO_IS_PRIVATE:\n",
    "        token = getpass.getpass(\"GitHub token: \" )\n",
    "        basic = base64.b64encode(f\"x-access-token:{token}\".encode()).decode()\n",
    "        command = [\"git\", \"-c\", f\"http.extraHeader=AUTHORIZATION: basic {basic}\", *command[1:]]\n",
    "    result = subprocess.run(command, text=True, capture_output=True)\n",
    "    if result.returncode != 0:\n",
    "        print(result.stdout)\n",
    "        print(result.stderr)\n",
    "        raise RuntimeError(\"GitHub 저장소 복제 실패\")\n",
    "    print(\"복제 완료:\", REPO_DIR)\n",
    "\n",
    "assert (REPO_DIR / \"config/experiments.json\").is_file(), \"실험 설정 파일이 없습니다.\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "\n",
    "install = [sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-r\", str(REPO_DIR / \"requirements.txt\"), \"-r\", str(REPO_DIR / \"requirements-dev.txt\")]\n",
    "if PRESET == \"all\":\n",
    "    install.extend([\"-r\", str(REPO_DIR / \"requirements-optional.txt\")])\n",
    "subprocess.run(install, check=True)\n",
    "print(\"패키지 설치 완료\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 데이터 확인·로컬 복사\n",
    "\n",
    "Drive의 원본은 유지하고 Colab 로컬 SSD로 복사합니다. 파일명은 정확히 `train.csv`, `test.csv`, `trackman_history.csv`여야 합니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import shutil\n",
    "\n",
    "required = [\"train.csv\", \"test.csv\", \"trackman_history.csv\"]\n",
    "LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)\n",
    "for name in required:\n",
    "    source = DRIVE_DATA_DIR / name\n",
    "    if not source.is_file():\n",
    "        raise FileNotFoundError(f\"못 찾음: {source}\")\n",
    "    destination = LOCAL_DATA_DIR / name\n",
    "    if (not destination.is_file() or destination.stat().st_size != source.stat().st_size or destination.stat().st_mtime_ns != source.stat().st_mtime_ns):\n",
    "        shutil.copy2(source, destination)\n",
    "    print(f\"{name}: {destination.stat().st_size / 1024**2:.1f} MB\")\n",
    "\n",
    "mapping = REPO_DIR / \"resources/pitcher_trackman_mapping.csv\"\n",
    "assert mapping.is_file(), mapping"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test_result = subprocess.run(\n",
    "    [sys.executable, \"-m\", \"pytest\", \"-q\"],\n",
    "    cwd=REPO_DIR,\n",
    "    text=True,\n",
    "    capture_output=True,\n",
    ")\n",
    "print(test_result.stdout)\n",
    "if test_result.returncode != 0:\n",
    "    print(test_result.stderr)\n",
    "    raise RuntimeError(\"단위 테스트 실패\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 실험 설정 확인·수정\n",
    "\n",
    "아래 셀에서 제공 피처 묶음과 모델을 확인합니다. 직접 실험을 추가하려면 `config['experiments']`에 항목을 추가하거나 기존 모델의 `params`를 수정한 뒤 셀을 실행하세요. 원본 설정 파일은 건드리지 않고 `/content/experiments_runtime.json`에 복사됩니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "from IPython.display import display\n",
    "\n",
    "config = json.loads((REPO_DIR / \"config/experiments.json\").read_text(encoding=\"utf-8\"))\n",
    "display(pd.DataFrame([\n",
    "    {\"feature_set\": name, \"blocks\": value.get(\"blocks\"), \"drop\": value.get(\"drop_patterns\"), \"description\": value.get(\"description\")}\n",
    "    for name, value in config[\"feature_sets\"].items()\n",
    "]))\n",
    "display(pd.DataFrame([\n",
    "    {\"model\": name, \"type\": value[\"type\"], \"description\": value.get(\"description\")}\n",
    "    for name, value in config[\"models\"].items()\n",
    "]))\n",
    "display(pd.DataFrame(config[\"experiments\"])[[\"name\", \"feature_set\", \"model\", \"presets\"]])\n",
    "\n",
    "# 예시: learning rate 수정\n",
    "# config[\"models\"][\"lgbm_base\"][\"params\"][\"learning_rate\"] = 0.02\n",
    "# config[\"models\"][\"lgbm_base\"][\"params\"][\"num_leaves\"] = 47\n",
    "\n",
    "RUNTIME_CONFIG = Path(\"/content/experiments_runtime.json\")\n",
    "RUNTIME_CONFIG.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding=\"utf-8\")\n",
    "print(\"실행 설정 저장:\", RUNTIME_CONFIG)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 실험 실행\n",
    "\n",
    "실행 로그를 그대로 출력합니다. 한 조합이 실패해도 오류가 해당 실행 폴더에 기록되고 다음 조합으로 넘어갑니다. 동일 데이터·동일 설정의 성공 실험은 자동으로 건너뜁니다."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "RESULT_DIR.mkdir(parents=True, exist_ok=True)\n",
    "command = [\n",
    "    sys.executable, \"-u\", \"-m\", \"src.experiment_runner\",\n",
    "    \"--config\", str(RUNTIME_CONFIG),\n",
    "    \"--train\", \"data/train.csv\",\n",
    "    \"--test\", \"data/test.csv\",\n",
    "    \"--trackman\", \"data/trackman_history.csv\",\n",
    "    \"--mapping\", \"resources/pitcher_trackman_mapping.csv\",\n",
    "    \"--output-dir\", str(RESULT_DIR),\n",
    "    \"--mode\", MODE,\n",
    "    \"--preset\", PRESET,\n",
    "    \"--validation-season\", \"2024\",\n",
    "    \"--n-jobs\", str(N_JOBS),\n",
    "]\n",
    "if ONLY_EXPERIMENTS:\n",
    "    command.extend([\"--only\", *ONLY_EXPERIMENTS])\n",
    "if RERUN:\n",
    "    command.append(\"--rerun\")\n",
    "\n",
    "process = subprocess.Popen(\n",
    "    command, cwd=REPO_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1\n",
    ")\n",
    "for line in process.stdout:\n",
    "    print(line, end=\"\")\n",
    "return_code = process.wait()\n",
    "if return_code != 0:\n",
    "    raise RuntimeError(f\"실험 실행 실패 (종료 코드 {return_code}). 위 로그와 runs/*/error.txt를 확인하세요.\")\n",
    "print(\"실험 완료:\", RESULT_DIR)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 결과 확인\n",
    "\n",
    "`brier_delta_vs_baseline`이 음수면 V1 LightGBM보다 개선된 것입니다. quick 결과를 확인한 뒤 첫 설정 셀에서 `MODE='full'`로 바꾸고 설정 셀부터 다시 실행하세요."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from IPython.display import Image, Markdown, display\n",
    "\n",
    "leaderboard = pd.read_csv(RESULT_DIR / \"leaderboard.csv\")\n",
    "display_columns = [\n",
    "    \"experiment\", \"feature_set\", \"model\", \"feature_count\",\n",
    "    \"brier\", \"brier_delta_vs_baseline\", \"brier_improvement_pct\",\n",
    "    \"auc\", \"logloss\", \"ece_10bin\", \"elapsed_seconds\"\n",
    "]\n",
    "display(leaderboard[[column for column in display_columns if column in leaderboard.columns]])\n",
    "\n",
    "for image_name in [\"leaderboard_brier.png\", \"improvement_vs_baseline.png\", \"score_vs_time.png\"]:\n",
    "    path = RESULT_DIR / image_name\n",
    "    if path.is_file():\n",
    "        display(Image(filename=str(path)))\n",
    "\n",
    "report_path = RESULT_DIR / \"experiment_report.md\"\n",
    "if report_path.is_file():\n",
    "    display(Markdown(report_path.read_text(encoding=\"utf-8\")))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 다음 실험 추가 방법\n",
    "\n",
    "1. quick starter 통과\n",
    "2. `MODE='full'`, `PRESET='starter'`로 피처 블록 효과 확정\n",
    "3. 가장 좋은 피처 묶음을 기준으로 `PRESET='extended'` 모델 비교\n",
    "4. XGBoost·CatBoost까지 필요할 때만 `PRESET='all'` 사용\n",
    "5. 새 피처는 `src/experiment_features.py`에 블록 함수로 추가하고 `config/experiments.json`에서 조합\n",
    "\n",
    "결과는 `experiment_history.csv`에 누적되고, 각 실행의 예측값·calibration·중요도·오류는 `runs/` 아래에 보존됩니다."
   ]
  }
 ],
 "metadata": {
  "colab": {
   "name": "LG_Aimers_피처_모델_자동실험실.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.x"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
