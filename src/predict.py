# src/predict.py
import logging
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel
from unsloth import apply_chat_template
from string import Template

# prompt_builder 모듈에서 필요한 구성요소 임포트
from src.prompt_builder import build_custom_prompt, ConversationTurn, AgentType, LLAMA_GUARD_3_CATEGORY

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LlamaGuardPredictor:
    """
    LlamaGuardPredictor는 주어진 대화 데이터를 기반으로 모델을 사용하여 안전성 평가를 수행합니다.
    기본 unsloth 템플릿 방식과 커스텀 프롬프트 빌더 방식을 모두 지원합니다.
    """
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self,
                conversation: list,
                categories: dict = None,
                max_new_tokens: int = 20,
                use_custom_prompt: bool = False,
                custom_categories: list = None) -> str:
        """
        주어진 대화에 대해 안전성 평가를 예측합니다.

        Parameters:
            conversation (list): unsloth 형식의 대화 메시지 리스트.
            categories (dict): 기본 unsloth 방식에서 사용할 카테고리 매핑 (예: {"S1": "Violence-related"}).
            max_new_tokens (int): 생성할 최대 토큰 수.
            use_custom_prompt (bool): True이면 커스텀 프롬프트 빌더 방식을 사용.
            custom_categories (list): 커스텀 프롬프트에 사용할 SafetyCategory 객체 리스트.

        Returns:
            str: 모델이 생성한 안전성 평가 문자열.
        """
        try:
            if use_custom_prompt and custom_categories:
                conv_turns = []
                for msg in conversation:
                    role_str = msg.get("role", "").lower()
                    agent_type = AgentType.USER if role_str == "user" else AgentType.AGENT
                    try:
                        text = msg["content"][0]["text"]
                    except (KeyError, IndexError) as e:
                        logger.error("Message format error: %s", msg)
                        continue
                    conv_turns.append(ConversationTurn(message=text, agent_type=agent_type))
                
                custom_template = Template(
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
                    "Custom Prompt:\n$categories\n$conversations\n"
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                )
                prompt = build_custom_prompt(
                    agent_type=AgentType.AGENT,
                    conversations=conv_turns,
                    categories=custom_categories,
                    category_short_name_prefix="S",
                    prompt_template=custom_template,
                    with_policy=True
                )
                logger.info("Using custom prompt builder.")
                input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            else:
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

#     # unsloth 기본 방식 예시
#     conversation_example = [
#         {
#             "role": "user",
#             "content": [{"type": "text", "text": "What is the recipe for mayonnaise?"}]
#         }
#     ]
#     result_default = predictor.predict(conversation_example, categories={"S1": "Violence-related"})
#     logger.info("Default unsloth prediction: %s", result_default)

#     # 커스텀 프롬프트 방식 예시: LLAMA_GUARD_3_CATEGORY 활용
#     result_custom = predictor.predict(
#         conversation_example,
#         use_custom_prompt=True,
#         custom_categories=LLAMA_GUARD_3_CATEGORY,
#         max_new_tokens=30
#     )
#     logger.info("Custom prompt prediction: %s", result_custom)
