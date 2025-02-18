from dataclasses import dataclass
from string import Template
from typing import List
from enum import Enum

class LlamaGuardVersion(Enum):
    LLAMA_GUARD_1 = "Llama Guard 1"
    LLAMA_GUARD_2 = "Llama Guard 2"
    LLAMA_GUARD_3 = "Llama Guard 3"

class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"

@dataclass
class SafetyCategory:
    name: str
    description: str

@dataclass
class ConversationTurn:
    message: str
    agent_type: AgentType

def build_custom_prompt(agent_type: AgentType, conversations: List[ConversationTurn],
                          categories: List[SafetyCategory],
                          category_short_name_prefix: str,
                          prompt_template: Template,
                          with_policy: bool = False) -> str:
    """
    Build a custom prompt for safety evaluation using the provided template.
    
    Parameters:
        agent_type (AgentType): The type of agent (User/Agent).
        conversations (List[ConversationTurn]): List of conversation turns.
        categories (List[SafetyCategory]): List of safety categories.
        category_short_name_prefix (str): Prefix for category numbering.
        prompt_template (Template): Template for prompt formatting.
        with_policy (bool): If True, include detailed category descriptions.
        
    Returns:
        str: The formatted prompt.
    """
    categories_str = "\n".join([
        f"{category_short_name_prefix}{i+1}: {c.name}" + (f"\n{c.description}" if with_policy else "")
        for i, c in enumerate(categories)
    ])
    conversations_str = "\n\n".join([f"{turn.agent_type.value}: {turn.message}" for turn in conversations])
    return prompt_template.substitute(agent_type=agent_type.value, categories=categories_str, conversations=conversations_str)
