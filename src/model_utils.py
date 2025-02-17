import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

def build_model(config):
    """
    사전학습된 모델과 토크나이저를 불러오고, 
    PEFT의 LoRA 구성을 적용하여 모델을 반환합니다.
    """
    logger.info(f"Loading tokenizer and model from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name)

    # LoRA 구성
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # 모델 구조에 맞게 조정 필요
    )
    logger.info("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    return model, tokenizer
