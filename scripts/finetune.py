#!/usr/bin/env python
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from src.prompt_builder import build_training_prompt, AgentType, ConversationTurn, load_safety_categories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(train_file: str, test_file: str, tokenizer):
    """
    학습 데이터를 불러오고, 각 예제에 대해 전체 시스템 프롬프트(안전 정책 + 전체 카테고리 목록)와
    사용자 대화(assistant 응답 포함)를 하나의 입력 텍스트로 구성합니다.
    단, 학습 목표는 어시스턴트 응답 부분에만 집중하도록 (train only on response) label 마스킹을 적용합니다.
    """
    dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})
    
    def preprocess_example(example):
        try:
            user_text = example["conversation"][0]["content"][0]["text"]
            assistant_text = example["conversation"][1]["content"][0]["text"]
        except (KeyError, IndexError):
            logger.error("Malformed conversation: %s", example)
            return example
        
        # 대화 목록 구성 (User 메시지와 Assistant 응답)
        conv_turns = [
            ConversationTurn(message=user_text, agent_type=AgentType.USER),
            ConversationTurn(message=assistant_text, agent_type=AgentType.AGENT)
        ]
        # safety_categories를 config로부터 로드
        safety_categories = load_safety_categories()
        # 전체 프롬프트 생성 (시스템 프롬프트 포함)
        full_prompt = build_training_prompt(
            agent_type=AgentType.AGENT,
            conversations=conv_turns,
            categories=safety_categories,
            category_short_name_prefix="S",
            with_policy=True
        )
        tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
        input_ids = tokenized["input_ids"]

        # "train only on response": 어시스턴트 응답 부분만 학습 대상으로 남김.
        # 어시스턴트 응답은 시스템 프롬프트 뒤, "<|start_header_id|>assistant<|end_header_id|>" 토큰 뒤에 위치합니다.
        assistant_delim = "<|start_header_id|>assistant<|end_header_id|>"
        delim_ids = tokenizer(assistant_delim, add_special_tokens=False)["input_ids"]

        # delim_ids가 input_ids 내에 존재하는 위치 찾기
        start_index = -1
        for i in range(len(input_ids) - len(delim_ids) + 1):
            if input_ids[i:i+len(delim_ids)] == delim_ids:
                start_index = i + len(delim_ids)
                break

        if start_index == -1:
            logger.error("Assistant delimiter not found in prompt: %s", full_prompt)
            # fallback: 전체 프롬프트를 학습 대상으로 사용 (권장하지 않음)
            labels = input_ids.copy()
        else:
            # assistant 응답 전까지는 -100 (loss 계산 제외)
            labels = [-100] * start_index + input_ids[start_index:]
            labels = labels[:len(input_ids)]  # 길이 맞추기

        example["input_ids"] = input_ids
        example["labels"] = labels
        return example

    dataset = dataset.map(preprocess_example)
    # 학습에 필요한 열만 남김 (input_ids, labels)
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["input_ids", "labels"]])
    dataset.set_format("torch")
    return dataset

def build_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="bfloat16",
        device_map="auto",
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer

def get_training_args(config):
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        fp16=config.use_fp16,
        logging_dir=f"{config.output_dir}/logs",
        logging_steps=50,
        ddp_find_unused_parameters=False,
    )
    return training_args

def main():
    from configs.finetune_config import FinetuneConfig
    config = FinetuneConfig()
    model, tokenizer = build_model(config)
    dataset = load_and_preprocess_data(config.train_file, config.test_file, tokenizer)
    training_args = get_training_args(config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()




        