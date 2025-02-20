import requests
import logging
from src.prompt_builder import load_safety_categories

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LlamaGuardClient:
    """
    LlamaGuardClient는 vLLM Serve의 OpenAI Compatible API를 호출하여,
    단일 텍스트 입력에 대한 Llama Guard 모델의 안전성 평가 결과(예: "safe", "unsafe S10")를 가져옵니다.
    
    입력 텍스트를 받으면 내부에서 시스템 프롬프트(안전 정책과 전체 카테고리 목록)를 포함한 메시지 리스트를 생성하며,
    각 메시지의 역할(role)이 번갈아 오도록 보장합니다.
    """
    def __init__(self, endpoint_url: str, model: str):
        """
        :param endpoint_url: vLLM Serve API 엔드포인트 (예: "http://localhost:8000")
        :param model: 사용할 모델 이름 (예: "meta-llama/Llama-Guard-3-1B")
        """
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = model

    def create_messages(self, input_text: str, use_system_prompt: bool = True) -> list:
        """
        단일 입력 텍스트를 기반으로 OpenAI API 형식의 메시지 리스트를 생성합니다.
        시스템 프롬프트가 포함되면, 안전 정책과 전체 카테고리 목록이 추가됩니다.
        """
        messages = []
        if use_system_prompt:
            safety_categories = load_safety_categories()
            categories_str = "\n".join([
                f"S{i+1}: {cat.name} - {cat.description}" for i, cat in enumerate(safety_categories)
            ])
            system_message = (
                "Classify the given prompt as safe or unsafe. For unsafe responses, specify the category (e.g., S1, S13) "
                "and provide reasoning behind your choice. Below is the safety policy with the following categories:\n"
                f"{categories_str}"
            )
            messages.append({"role": "system", "content": system_message})
        # 사용자 메시지: 'User:' 접두사를 명시적으로 포함하여 원본 형식을 반영
        user_message = {"role": "user", "content": f"User: {input_text}"}
        messages.append(user_message)
        return messages

    def fix_alternating_roles(self, messages: list) -> list:
        """
        메시지 리스트에서 역할(role)이 번갈아 오도록 보장합니다.
        만약 연속된 메시지의 role이 동일하다면, 현재 메시지의 role을 자동으로 반대 역할로 수정합니다.
        """
        if not messages:
            return messages
        fixed = [messages[0]]
        for msg in messages[1:]:
            prev_role = fixed[-1].get("role", "").lower()
            curr_role = msg.get("role", "").lower()
            if curr_role == prev_role:
                new_role = "assistant" if prev_role == "user" else "user"
                logger.warning(
                    "Non-alternating roles detected; changing role from '%s' to '%s' in message: %s",
                    curr_role, new_role, msg
                )
                msg["role"] = new_role
            fixed.append(msg)
        return fixed

    def chat(self, input_text: str, max_new_tokens: int = 20, temperature: float = 0.7,
             use_system_prompt: bool = True) -> str:
        """
        vLLM Serve의 OpenAI Compatible API에 단일 텍스트 입력에 대한 채팅 요청을 보냅니다.
        
        :param input_text: 평가할 단일 텍스트 입력
        :param max_new_tokens: 생성할 최대 토큰 수
        :param temperature: 샘플링 온도
        :param use_system_prompt: 시스템 프롬프트(안전 정책 포함) 사용 여부
        :return: 모델이 생성한 평가 결과 문자열 (예: "safe" 또는 "unsafe S10")
        """
        messages = self.create_messages(input_text, use_system_prompt)
        messages = self.fix_alternating_roles(messages)
        payload = {
            "model": self.model,
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature
        }
        logger.info("Sending payload to vLLM Serve: %s", payload)
        response = requests.post(f"{self.endpoint_url}/v1/chat/completions", json=payload)
        if response.status_code != 200:
            logger.error("Error from server: %s", response.text)
            response.raise_for_status()
        result = response.json()
        # OpenAI API 형식: {"choices": [{"message": {"role": "assistant", "content": "응답내용"}}], ...}
        return result["choices"][0]["message"]["content"]

