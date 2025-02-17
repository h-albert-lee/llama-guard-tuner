import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

def load_and_preprocess_data(train_file: str, test_file: str, tokenizer):
    """
    JSONL 파일로부터 데이터를 로드하고,
    - label 문자열("safe", "unsafe")를 정수 인코딩 (safe: 1, unsafe: 0)
    - tokenizer를 적용하여 입력 텐서를 생성합니다.
    """
    logger.info("Loading dataset...")
    dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})
    
    # label을 정수로 변환
    def preprocess_labels(example):
        example["label"] = 1 if example["label"].lower() == "safe" else 0
        return example

    dataset = dataset.map(preprocess_labels)

    # 텍스트 토크나이즈
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

    logger.info("Tokenizing dataset...")
    dataset = dataset.map(tokenize_function, batched=True)
    # Trainer에서 사용하기 위해 필요한 열만 남김
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["input_ids", "attention_mask", "label"]])
    dataset.set_format("torch")
    
    return dataset
