# src/prompt_builder.py
import logging
from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum

logger = logging.getLogger(__name__)

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

@dataclass
class SafetyCategory:
    name: str
    description: str

# 간단하게 구성한 대화 템플릿 (User/Agent 형식)
# 시스템 프롬프트는 안전 정책과 전체 카테고리 목록을 포함합니다.
SYSTEM_MESSAGE = (
    "Classify the given prompt as safe or unsafe. For unsafe responses, specify the category (e.g., S1, S13) "
    "and provide reasoning behind your choice. Below is the safety policy with the following categories:"
)

# 시스템 프롬프트 템플릿: 안전 정책과 카테고리 목록을 포함합니다.
SYSTEM_PROMPT_TEMPLATE = Template(
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "$system_message\n"
    "$categories\n"
    "<|eot_id|>\n"
)

# 대화 템플릿: User와 Agent 메시지를 명시적으로 구분합니다.
CONVERSATION_TEMPLATE = Template(
    "User: $user_text\n"
    "Agent: $agent_text"
)

def load_safety_categories() -> List[SafetyCategory]:
    try:
        from configs.safety_categories import SAFETY_CATEGORIES_CONFIG
        return [SafetyCategory(**cat) for cat in SAFETY_CATEGORIES_CONFIG]
    except Exception as e:
        logger.exception("Failed to load safety categories: %s", e)
        raise

def build_training_prompt(user_text: str, agent_text: str, categories: List[SafetyCategory],
                          category_short_name_prefix: str = "S", with_policy: bool = True) -> str:
    """
    학습용 전체 프롬프트를 구성합니다.
    - 시스템 프롬프트: 안전 정책 및 전체 카테고리 목록(각 카테고리는 "S번호: 이름" 형식으로 표시됨)
    - 대화: "User: ..."와 "Agent: ..." 형식으로 구성
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
    conversation_part = CONVERSATION_TEMPLATE.substitute(
        user_text=user_text,
        agent_text=agent_text
    )
    full_prompt = system_part + conversation_part
    return full_prompt
