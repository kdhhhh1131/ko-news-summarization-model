**# ko-news-summarization-model
**
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
* **Finetuned Model**: [`kdhhhh1131/ko-news-summarization-model`](https://huggingface.co/kdhhhh1131/ko-news-summarization-model)
<br/>

## 💻 설치 및 실행 방법

### 1. 환경 구성 (Colab 권장)
본 모델은 4-bit 양자화(Quantization)된 베이스 모델 위에 학습된 LoRA 가중치를 병합하여 사용하는 방식으로 동작합니다. Google Colab(무료 T4 GPU) 환경에서 가장 원활하게 실행할 수 있습니다.
```bash
### 1. 환경 설정 및 필수 라이브러리 설치
모델 로드 및 4-bit 추론을 위해 아래 라이브러리들을 설치해야 합니다.
```bash
pip install -q transformers accelerate torch peft torchao
pip install -U bitsandbytes>=0.46.1

```


### 🔍 핵심 코드 설명 (Code Breakdown)

추론 코드에서 모델의 성능을 최적화하고 정확한 결과를 얻기 위해 적용된 주요 핵심 로직은 다음과 같습니다.

* **LoRA 어댑터 병합 (`PeftModel.from_pretrained`)**
  본 저장소의 모델 파일은 전체 가중치가 아닌 학습된 파라미터만 담긴 **LoRA 어댑터**입니다. 따라서 4-bit로 양자화된 가벼운 베이스 모델을 먼저 메모리에 올린 뒤, 그 위에 어댑터를 덧씌우는(Merge) 방식을 사용하여 VRAM 사용량을 획기적으로 절약합니다.


## 📊 모델 성능 비교 및 분석 (Model Comparison)

베이스 모델(왼쪽). 파인튜닝 모델(오른쪽)
<img width="1383" height="586" alt="스크린샷 2026-06-21 131750" src="https://github.com/user-attachments/assets/bc635bf7-1ff2-4048-9e34-75c28c8ccf52" />
<img width="1367" height="519" alt="스크린샷 2026-06-21 131806" src="https://github.com/user-attachments/assets/f67248ff-bd43-449c-b48f-91dbf602b9af" />

동일한 경제 뉴스 기사를 입력했을 때, 파인튜닝 전(베이스 모델)과 후(News-Summary-Bot)의 출력 결과를 비교한 분석입니다.

### 1. 베이스 모델 (Qwen2.5-1.5B-Instruct)
> **출력 결과:** > 청년층과 고령층의 고용 상황이 서로 다르게 변화하고 있습니다. 청년층의 상용직은 감소 속도가 빠르게 나타나고, 고령층은 증가세를 이어가고 있습니다. 이러한 변화는 청년층 고용 둔화와 고령층 노동시장 유입 확대가 맞물린 결과로 보입니다... (중략) ...'K-뉴딜 아카데미' 등 기존 청년 고용 지원 사업 외에도 신규 대책을 발굴하기로 했습니다.

* **📉 한계점 분석:**
  * **지시사항 무시:** '3줄 요약'이라는 명확한 제약 조건을 무시하고, 일반적인 산문 형태로 길게 서술했습니다.
  * **정보 압축 실패:** 기사의 흐름을 부드럽게 이어가는 데 치중하여, 정작 독자가 가장 궁금해할 구체적인 '수치 데이터'는 모두 누락되었습니다.

### 2. 파인튜닝 모델 (News-Summary-Bot)
> **출력 결과:**
> 고용 안정성이 높은 상용직 근로자에서 60세 이상 고령층이 청년층을 앞질렀다. 올해 60세 이상 상용직은 220만명, 청년층은 212만4천명이었다. 청년층 상용직은 4년 연속 줄고 있으며, 청년층 상용직은 인구보다 더 빠른 속도로 감소했다. 고령층 상용직은 집계 이래 증가세를 이어오고 있다.

* **📈 개선점 및 성능 분석:**
  * **완벽한 형식 통제:** 군더더기나 감정적인 배경 설명을 완벽히 걷어내고, 지시사항에 맞춰 문장을 짧고 간결하게 압축했습니다.
  * **정확한 정보 추출 (Information Extraction):** 기사 내의 핵심 통계인 `220만 명`, `212만 4천 명`, `4년 연속` 등의 핵심 팩트를 누락 없이 정확하게 포착하여 요약문에 반영했습니다.
  * **결론:** 파인튜닝을 통해 모델이 뉴스 도메인에서 **"독자가 본문을 읽지 않아도 핵심 지표를 즉각적으로 파악할 수 있도록 돕는"** 실용적인 요약 도구로 확실하게 진화했음을 보여줍니다.

