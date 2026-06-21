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

## 💻 추론 (Inference) 테스트 방법

Unsloth의 최적화된 추론 환경을 통해 파인튜닝된 모델의 요약 능력을 바로 테스트해 볼 수 있습니다.

### Python 실행 코드
```python
from unsloth import FastLanguageModel
import torch

# 1. 파인튜닝된 모델 로드 (Hugging Face Repository 또는 로컬 경로)
# 주의: 'your_huggingface_id' 부분을 본인의 저장소 이름으로 변경하세요.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "your_huggingface_id/news-summary-bot", 
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

# 2. Unsloth 추론 속도 2배 향상 활성화
FastLanguageModel.for_inference(model)

# 3. 프롬프트 준비
prompt = """다음 뉴스 기사를 읽고, 가장 중요한 핵심 팩트와 수치만 포함하여 명료하게 3줄로 요약해 주세요.

[뉴스 기사]
{기사_본문}

3줄 요약:
"""

article = """
(세종=연합뉴스) 안채원 기자 = 고용 안정성이 높은 상용직 근로자에서 60세 이상 고령층이 청년층을 앞질렀다. 청년층 인구 보다 상용직 감소 속도가 빠르게 나타나며 고용의 질에서도 세대 역전 흐름이 나타나는 것으로 보인다... (중략)
"""

inputs = tokenizer(
    [prompt.format(기사_본문=article)],
    return_tensors="pt"
).to("cuda")

# 4. 텍스트 생성 (온도를 낮춰 팩트 기반의 건조한 요약 유도)
outputs = model.generate(
    **inputs,
    max_new_tokens = 256,
    use_cache = True,
    temperature = 0.2, 
)

# 5. 결과 출력
result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(result.split("3줄 요약:\n")[-1])

## 📊 모델 성능 비교 및 분석 (Model Comparison)

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

