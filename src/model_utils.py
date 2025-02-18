import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)

def build_model(config):
    """
    사전학습된 LLaMA Guard 모델과 토크나이저를 불러오고,  
    PEFT(LoRA)를 적용하여 모델을 반환합니다.
    
    - AutoModelForCausalLM 사용
    - torch_dtype는 bfloat16, device_map은 "auto" 설정
    """
    logger.info(f"Loading tokenizer and model from {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # LoRA 구성 (모델 아키텍처에 맞게 target_modules 조정)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    logger.info("Applying LoRA configuration...")
    model = get_peft_model(model, lora_config)
    return model, tokenizer
