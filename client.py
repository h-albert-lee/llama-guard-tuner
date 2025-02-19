import requests
import logging
from src.prompt_builder import load_safety_categories

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LlamaGuardClient:
    """
    LlamaGuardClient는 vLLM Serve의 OpenAI Compatible API를 호출하여,
    Llama Guard 모델의 안전성 평가 결과를 가져옵니다.
    
    필요에 따라 시스템 프롬프트(안전 정책 및 전체 카테고리 목록)를 포함한 메시지 리스트를 생성할 수 있습니다.
    """
    def __init__(self, endpoint_url: str, model: str):
        """
        :param endpoint_url: vLLM Serve API 엔드포인트 (예: "http://localhost:8000")
        :param model: 사용 모델 이름 (예: "meta-llama/Llama-Guard-3-1B")
        """
        self.endpoint_url = endpoint_url.rstrip("/")
        self.model = model

    def create_messages(self, conversation: list, use_system_prompt: bool = True) -> list:
        """
        OpenAI API 형식의 메시지 리스트를 생성합니다.
        :param conversation: [{"role": "user", "content": "메시지 내용"}] 형식의 대화 리스트
        :param use_system_prompt: True인 경우, 시스템 프롬프트를 첫번째 메시지로 추가합니다.
        :return: 메시지 리스트
        """
        messages = []
        if use_system_prompt:
            # 시스템 프롬프트에 safety category config를 포함
            safety_categories = load_safety_categories()
            # 간단히 각 카테고리를 "S번호: 이름 - 설명" 형식의 문자열로 나열
            categories_str = "\n".join([
                f"S{i+1}: {cat.name} - {cat.description}" for i, cat in enumerate(safety_categories)
            ])
            system_message = (
                "Classify the given prompt as safe or unsafe. For unsafe responses, specify the category (e.g., S1, S13) "
                "and provide reasoning behind your choice. Below is the safety policy with the following categories:\n"
                f"{categories_str}"
            )
            messages.append({"role": "system", "content": system_message})
        messages.extend(conversation)
        return messages

    def chat(self, conversation: list, max_new_tokens: int = 20, temperature: float = 0.7,
             use_system_prompt: bool = True) -> str:
        """
        vLLM Serve의 OpenAI Compatible API에 채팅 요청을 보냅니다.
        :param conversation: [{"role": "user", "content": "메시지 내용"}] 형식의 대화 리스트
        :param max_new_tokens: 생성할 최대 토큰 수
        :param temperature: 샘플링 온도
        :param use_system_prompt: 시스템 프롬프트 포함 여부
        :return: 어시스턴트가 생성한 응답 문자열
        """
        messages = self.create_messages(conversation, use_system_prompt=use_system_prompt)
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
        # OpenAI API 형식의 응답을 가정:
        # {"choices": [{"message": {"role": "assistant", "content": "응답내용"}}], ...}
        return result["choices"][0]["message"]["content"]