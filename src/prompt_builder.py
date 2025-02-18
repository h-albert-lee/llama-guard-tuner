import logging
from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum

# 로깅 설정
logger = logging.getLogger(__name__)

# 에이전트 유형 정의
class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

# SafetyCategory와 ConversationTurn 데이터 클래스
@dataclass
class SafetyCategory:
    name: str
    description: str

@dataclass
class ConversationTurn:
    message: str
    agent_type: AgentType

# --- 시스템 프롬프트 (안전 정책) ---
SYSTEM_MESSAGE = (
    "Classify the given prompt as safe or unsafe. For unsafe responses, specify the category (e.g., S1, S13) "
    "and provide reasoning behind your choice. Below is the safety policy with the following categories:"
)

# 프롬프트 템플릿 (시스템 프롬프트 포함)
SYSTEM_PROMPT_TEMPLATE = Template(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "$system_message\n"
    "$categories\n"
    "<|eot_id|>\n"
)

# 사용자 및 어시스턴트 대화 템플릿
CONVERSATION_TEMPLATE = Template(
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
    "$conversations\n"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
)

def load_safety_categories() -> List[SafetyCategory]:
    """
    configs/safety_categories.py에 정의된 config를 읽어 SafetyCategory 객체 리스트로 반환합니다.
    """
    try:
        from configs.safety_categories import SAFETY_CATEGORIES_CONFIG
        return [SafetyCategory(**cat) for cat in SAFETY_CATEGORIES_CONFIG]
    except Exception as e:
        logger.exception("Failed to load safety categories: %s", e)
        raise

def build_training_prompt(agent_type: AgentType,
                            conversations: List[ConversationTurn],
                            categories: List[SafetyCategory],
                            category_short_name_prefix: str,
                            with_policy: bool = True) -> str:
    """
    학습용 전체 프롬프트를 구성
    시스템 프롬프트에는 고정된 안전 정책과 전체 카테고리 목록이 포함되고,
    이어서 사용자 대화가 포함
    """
    categories_str = "\n".join([
        f"{category_short_name_prefix}{i+1}: {c.name}" +
        (f"\n{c.description}" if with_policy else "")
        for i, c in enumerate(categories)
    ])
    system_part = SYSTEM_PROMPT_TEMPLATE.substitute(
        system_message=SYSTEM_MESSAGE,
        categories=categories_str
    )
    conversations_str = "\n\n".join([
        f"{turn.agent_type.value}: {turn.message}" for turn in conversations
    ])
    conversation_part = CONVERSATION_TEMPLATE.substitute(
        conversations=conversations_str
    )
    full_prompt = system_part + conversation_part
    return full_prompt

def build_custom_prompt(agent_type: AgentType,
                          conversations: List[ConversationTurn],
                          categories: List[SafetyCategory],
                          category_short_name_prefix: str,
                          prompt_template: Template,
                          with_policy: bool = False) -> str:
    """
    커스텀 프롬프트 빌더 (시스템 프롬프트 미포함) - 주로 추론 단계에서 unsloth 방식과 비교용으로 사용.
    """
    try:
        categories_str = "\n".join([
            f"{category_short_name_prefix}{i+1}: {c.name}" +
            (f"\n{c.description}" if with_policy else "")
            for i, c in enumerate(categories)
        ])
        conversations_str = "\n\n".join([
            f"{turn.agent_type.value}: {turn.message}" for turn in conversations
        ])
        prompt = prompt_template.substitute(
            agent_type=agent_type.value,
            categories=categories_str,
            conversations=conversations_str
        )
        return prompt
    except Exception as e:
        logger.exception("Failed to build custom prompt: %s", e)
        raise
