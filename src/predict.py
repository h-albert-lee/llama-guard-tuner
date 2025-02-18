import logging
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from unsloth import apply_chat_template  # 기본 unsloth 방식 지원
from string import Template
from src.prompt_builder import (
    build_custom_prompt,
    build_training_prompt,
    ConversationTurn,
    AgentType,
    load_safety_categories
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LlamaGuardPredictor:
    """
    LlamaGuardPredictor는 시스템 프롬프트(안전 정책 포함)와 대화 템플릿을 사용하여,
    모델로부터 안전성 평가(예: "safe" 또는 "unsafe S1", "unsafe S13" 등)를 생성
    
    - 학습 시: 전체 시스템 프롬프트 + 사용자 대화 → 모델 fine-tuning
    - 추론 시: unsloth 기본 방식 또는 커스텀 프롬프트(옵션) 사용
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # config로부터 safety category 리스트 로드
        self.safety_categories = load_safety_categories()

    def predict(self,
                conversation: list,
                unsloth_categories: dict = None,
                max_new_tokens: int = 20,
                use_custom_prompt: bool = False) -> str:
        """
        주어진 대화에 대해 안전성 평가를 예측합니다.
        
        Parameters:
            conversation (list): unsloth 형식의 대화 메시지 리스트.
            unsloth_categories (dict): 기본 unsloth 방식에서 사용할 카테고리 매핑.
            max_new_tokens (int): 생성할 최대 토큰 수.
            use_custom_prompt (bool): True이면 커스텀 프롬프트 빌더(시스템 프롬프트 포함)를 사용.
        
        Returns:
            str: 모델이 생성한 안전성 평가 문자열.
        """
        try:
            if use_custom_prompt:
                conv_turns = []
                for msg in conversation:
                    role_str = msg.get("role", "").lower()
                    agent_type = AgentType.USER if role_str == "user" else AgentType.AGENT
                    try:
                        text = msg["content"][0]["text"]
                    except (KeyError, IndexError):
                        logger.error("Message format error: %s", msg)
                        continue
                    conv_turns.append(ConversationTurn(message=text, agent_type=agent_type))
                
                # 커스텀 프롬프트 방식: 시스템 프롬프트(안전 정책 포함) + 대화
                prompt = build_training_prompt(
                    agent_type=AgentType.AGENT,
                    conversations=conv_turns,
                    categories=self.safety_categories,
                    category_short_name_prefix="S",
                    with_policy=True
                )
                logger.info("Using custom training prompt (with system safety policy).")
                input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            else:
                logger.info("Using default unsloth chat template.")
                input_ids = self.tokenizer.apply_chat_template(
                    conversation,
                    return_tensors="pt",
                    categories=unsloth_categories,
                ).to(self.model.device)
            
            prompt_len = input_ids.shape[1]
            self.model.eval()
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=0,
                )
            generated_tokens = output[:, prompt_len:]
            result = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            return result
        except Exception as e:
            logger.exception("Error during prediction: %s", e)
            raise

# # 프로덕션 환경에서의 사용 예시
# if __name__ == "__main__":
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     import torch

#     model_id = "meta-llama/Llama-Guard-3-1B"
#     try:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id,
#             torch_dtype=torch.bfloat16,
#             device_map="auto",
#         )
#         tokenizer = AutoTokenizer.from_pretrained(model_id)
#     except Exception as e:
#         logger.exception("Failed to load model/tokenizer: %s", e)
#         raise

#     predictor = LlamaGuardPredictor(model, tokenizer)

#     conversation_example = [
#         {
#             "role": "user",
#             "content": [{"type": "text", "text": "What is the recipe for mayonnaise?"}]
#         }
#     ]
#     # 기본 unsloth 방식 예시
#     result_default = predictor.predict(conversation_example, unsloth_categories={"S1": "Violence-related"})
#     logger.info("Default unsloth prediction: %s", result_default)

#     # 커스텀 프롬프트 방식 예시 (시스템 프롬프트 포함, safety category config 사용)
#     result_custom = predictor.predict(
#         conversation_example,
#         use_custom_prompt=True,
#         max_new_tokens=30
#     )
#     logger.info("Custom prompt prediction: %s", result_custom)
