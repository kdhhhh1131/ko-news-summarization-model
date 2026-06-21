# ko-news-summarization-model

# 📰 News-Summary-Bot (뉴스 3줄 요약 봇)

긴 뉴스 기사를 입력하면 글의 핵심 정보와 수치를 파악하여 **명료하게 3줄로 요약해 주는 경량화 언어 모델(SLM)** 입니다. 

방대한 텍스트 속에서 인과관계나 군더더기 서술은 과감히 생략하고, 가장 중요한 '팩트(Fact)'와 '수치'만을 남겨 정보의 압축률을 극대화하도록 파인튜닝 되었습니다.

<br/>

## 🎯 주요 기능 및 파인튜닝 효과
* **압도적인 요약 압축률**: 베이스 모델은 글의 흐름을 살려 길게 설명하려는 경향(장황함)이 있고, 때로는 정보의 왜곡이 발생하기도 합니다. 파인튜닝된 본 모델은 지시사항(3줄 요약)을 엄격하게 따르며 불필요한 배경 설명을 날리고 핵심만 요약합니다.
* **정확한 정보 추출**: 뉴스 기사 내의 구체적인 통계, 수치, 장소 등의 팩트를 잃지 않고 정확하게 요약문에 반영합니다.

<br/>

## 🛠 사용한 기술 스택 (Tech Stack)
* **Base Model**: [`Qwen/Qwen2.5-1.5B-Instruct-bnb-4bit`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
* **Dataset**: [`daekeun-ml/naver-news-summarization-ko`](https://huggingface.co/datasets/daekeun-ml/naver-news-summarization-ko)
* **Framework / Tool**: `Unsloth`, `QLoRA`, `Hugging Face Transformers`

<br/>

## 🧠 파인튜닝 설계 설명

### 1. 모델 및 데이터 선택 근거
* **모델 선택 (Qwen2.5-1.5B)**: 1.5B라는 초경량 사이즈임에도 불구하고 한국어 토큰 효율과 문해력이 매우 뛰어납니다. 특히 `bnb-4bit` 양자화 모델을 활용하여 무료 Colab 환경(T4 GPU)에서도 VRAM 부족 현상 없이 원활하고 빠른 학습이 가능하도록 설계했습니다.
* **데이터셋 선택**: 네이버 뉴스 본문과 전문가가 요약한 고품질 요약문이 쌍(Pair)으로 이루어져 있어, 모델에게 '불필요한 문장 쳐내기'와 '핵심 압축'의 패턴을 가르치기에 가장 이상적인 데이터라고 판단했습니다.

### 2. 하이퍼파라미터 (Hyperparameters)
빠른 수렴과 메모리 효율을 위해 Unsloth 환경에서 다음과 같이 QLoRA 설정을 적용했습니다.
* **Epochs**: 30
* **Context Length**: 4096
* **Learning Rate**: 1e-5 (0.00001)
* **LoRA Rank (r)**: 4
* **LoRA Alpha**: 8
* **Dropout**: 0.0
* **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

<br/>

## 💻 설치 및 실행 방법

### 1. 환경 구성 (Colab 권장)
본 프로젝트는 Google Colab T4 (무료 버전) 환경에서 원활하게 동작합니다.
```bash
# Unsloth 및 필수 라이브러리 설치
!pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
