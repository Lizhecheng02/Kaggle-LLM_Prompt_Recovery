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
kaggle datasets download -d xuanmingzhang777/gemini-dataset-3-8-k
unzip gemini-dataset-3-8-k.zip
kaggle datasets download -d lizhecheng/lzc-llm-prompt-recovery-dataset
unzip lzc-llm-prompt-recovery-dataset.zip
kaggle datasets download -d lizhecheng/llm-prompt-recovery-extra-dataset
llm-prompt-recovery-extra-dataset.zip
kaggle competitions download -c llm-prompt-recovery
unzip llm-prompt-recovery.zip
```
