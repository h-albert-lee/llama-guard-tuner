import logging
from datasets import load_dataset
from unsloth import standardize_sharegpt, apply_chat_template

logger = logging.getLogger(__name__)

def load_and_preprocess_data(train_file: str, test_file: str, tokenizer):
    """
    1. ShareGPT 형식의 데이터를 로드하고 standardize_sharegpt()로 표준화합니다.
    2. unsloth의 apply_chat_template()를 이용해 챗 템플릿을 적용합니다.
       - default_system_message는 "Classify the given prompt as malicious or not malicious. Also give a reasoning behind your choice."
    3. 응답(assistant) 부분만 추출하여 'text' 필드에 덮어씁니다.
    4. tokenizer를 적용하여 텐서로 변환합니다.
    
    *LLaMA Guard는 train on responses only 방식으로 파인튜닝합니다.*
    """
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})
    
    # 1. 표준화 (ShareGPT)
    logger.info("Standardizing ShareGPT dataset...")
    dataset = standardize_sharegpt(dataset)
    
    # 2. 챗 템플릿 적용
    chat_template = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "{INPUT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        "{OUTPUT}<|eot_id|>"
    )
    logger.info("Applying chat template...")
    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template,
        default_system_message="""Provide your safety assessment for $agent_type in the above conversation:
        - First line must read 'safe' or 'unsafe'.
        - If unsafe, a second line must include a comma-separated list of violated categories""",
    )
    
    # 3. 응답(assistant) 부분만 추출
    def extract_assistant_response(example):
        text = example["text"]
        start_token = "<|start_header_id|>assistant<|end_header_id|>"
        end_token = "<|eot_id|>"
        start_idx = text.find(start_token)
        if start_idx != -1:
            start_idx += len(start_token)
            end_idx = text.find(end_token, start_idx)
            if end_idx != -1:
                # assistant 응답만 추출하여 text 필드에 저장
                example["text"] = text[start_idx:end_idx].strip()
        return example

    logger.info("Extracting assistant responses only...")
    dataset = dataset.map(extract_assistant_response)
    
    # 4. Tokenization (max_length 조정 가능)
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    
    logger.info("Tokenizing dataset...")
    dataset = dataset.map(tokenize_function, batched=True)
    
    # LM fine-tuning을 위해 'input_ids'와 'attention_mask'만 유지
    keep_cols = ["input_ids", "attention_mask"]
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in keep_cols])
    dataset.set_format("torch")
    
    return dataset
