## This Repo is for [Kaggle - LLM Prompt Recovery](https://www.kaggle.com/competitions/llm-prompt-recovery)

### Python Environment

#### 1. Install Packages

```b
pip install -r requirements.txt
```

### Prepare Data

#### 1. Set Kaggle Api

```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_api_key"
```

#### 2. Download Dataset
```bash
cd datasets
kaggle datasets download -d lizhecheng/lzc-llm-prompt-recovery-dataset
unzip lzc-llm-prompt-recovery-dataset.zip
kaggle datasets download -d lizhecheng/llm-prompt-recovery-extra-dataset
unzip llm-prompt-recovery-extra-dataset.zip
kaggle competitions download -c llm-prompt-recovery
unzip llm-prompt-recovery.zip
```

### Train Model
You can use different ``ipynb`` or ``py`` files to do it. (However, there is a problem with very huge loss while fine-tuning ``gemma-7b``)
