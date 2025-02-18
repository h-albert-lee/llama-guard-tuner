import logging
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from unsloth import apply_chat_template
from string import Template

# 새로 추가된 프롬프트 빌더 모듈 임포트
from src.prompt_builder import build_custom_prompt, SafetyCategory, ConversationTurn, AgentType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LlamaGuardPredictor:
    """
    LlamaGuardPredictor는 모델과 토크나이저를 활용하여 안전성 평가를 위한
    추론을 수행합니다. 기본 unsloth 템플릿 방식과 커스텀 프롬프트 빌더 방식을 모두 지원합니다.
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize the predictor with a model and a tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, conversation: list, categories: dict = None, max_new_tokens: int = 20, 
                use_custom_prompt: bool = False, custom_categories: list = None) -> str:
        """
        Predict safety assessment for the given conversation.
        
        Parameters:
            conversation (list): 대화 메시지 리스트 (각 항목은 unsloth 형식의 dict).
                예시:
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What is the recipe for mayonnaise?"}]
                    }
                ]
            categories (dict): 기본 unsloth 방식에서 사용할 카테고리 매핑 (예: {"S1": "My custom category"}).
            max_new_tokens (int): 생성할 최대 토큰 수.
            use_custom_prompt (bool): True이면 커스텀 프롬프트 빌더를 사용.
            custom_categories (list): 커스텀 프롬프트에 사용할 SafetyCategory 객체 목록.
        
        Returns:
            str: 모델 출력(예: "safe" 또는 "unsafe S1, S13" 등).
        """
        try:
            if use_custom_prompt and custom_categories:
                # unsloth 형식의 conversation을 ConversationTurn 객체 목록으로 변환
                conv_turns = []
                for msg in conversation:
                    role_str = msg.get("role", "").lower()
                    if role_str == "user":
                        agent_type = AgentType.USER
                    elif role_str == "assistant":
                        agent_type = AgentType.AGENT
                    else:
                        agent_type = AgentType.USER  # 기본값
                    try:
                        text = msg["content"][0]["text"]
                    except (KeyError, IndexError) as e:
                        logger.error("Message format error in conversation: %s", msg)
                        continue
                    conv_turns.append(ConversationTurn(message=text, agent_type=agent_type))
                
                # 커스텀 프롬프트 빌더를 사용하여 프롬프트 생성
                custom_template = Template(
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                    "Custom Prompt:\n$categories\n$conversations\n"
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
                prompt = build_custom_prompt(
                    agent_type=AgentType.AGENT,
                    conversations=conv_turns,
                    categories=custom_categories,
                    category_short_name_prefix="S",  # 필요에 따라 조정
                    prompt_template=custom_template,
                    with_policy=True
                )
                logger.info("Using custom prompt builder.")
                input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            else:
                # 기본 unsloth 템플릿 방식 사용
                logger.info("Using default unsloth chat template.")
                input_ids = self.tokenizer.apply_chat_template(
                    conversation,
                    return_tensors="pt",
                    categories=categories,
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

# # 예시: 프로덕션 환경에서의 사용법
# if __name__ == "__main__":
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     import torch

#     # 모델과 토크나이저 로드 (예: Llama-Guard-3-1B)
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

#     # 기본 unsloth 방식 예시
#     conversation_example = [
#         {
#             "role": "user",
#             "content": [{"type": "text", "text": "What is the recipe for mayonnaise?"}]
#         }
#     ]
#     result_default = predictor.predict(conversation_example, categories={"S1": "Violence-related"})
#     logger.info("Default unsloth prediction: %s", result_default)

#     # 커스텀 프롬프트 방식 예시
#     # 새 안전 카테고리 예시 (S13: 개인정보 탐지)
#     custom_categories = [
#         SafetyCategory(
#             name="Personal Data Extraction",
#             description="Should not provide information that enables the extraction of personal data."
#         ),
#         SafetyCategory(
#             name="Violence and Hate",
#             description="Should not enable, encourage, or provide instructions for violence or hate."
#         )
#     ]
#     result_custom = predictor.predict(
#         conversation_example,
#         use_custom_prompt=True,
#         custom_categories=custom_categories,
#         max_new_tokens=30
#     )
#     logger.info("Custom prompt prediction: %s", result_custom)
