#!/usr/bin/env python
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from src.prompt_builder import build_training_prompt, AgentType, load_safety_categories
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(train_file: str, test_file: str, tokenizer):
    """
    학습 데이터를 불러오고, 각 예제에 대해 전체 프롬프트를 구성합니다.
    전체 프롬프트는 시스템 프롬프트(안전 정책 및 전체 카테고리 목록)와
    대화 부분("User: ...\nAgent: ...")으로 이루어집니다.
    단, "train only on response" 전략을 적용해, 어시스턴트 응답 부분만을 학습 대상으로 합니다.
    """
    dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})
    
    def preprocess_example(example):
        try:
            user_text = example["conversation"][0]["content"][0]["text"]
            agent_text = example["conversation"][1]["content"][0]["text"]
        except (KeyError, IndexError):
            logger.error("Malformed conversation: %s", example)
            return example
        
        safety_categories = load_safety_categories()
        full_prompt = build_training_prompt(
            user_text=user_text,
            agent_text=agent_text,
            categories=safety_categories,
            category_short_name_prefix="S",
            with_policy=True
        )
        tokenized = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)
        input_ids = tokenized["input_ids"]

        # "train only on response": 어시스턴트 응답 부분만 학습 대상(label)으로 남김.
        # 어시스턴트 응답은 대화 템플릿에서 "Agent:"로 시작합니다.
        agent_delim = "Agent:"
        delim_ids = tokenizer(agent_delim, add_special_tokens=False)["input_ids"]

        start_index = -1
        for i in range(len(input_ids) - len(delim_ids) + 1):
            if input_ids[i:i+len(delim_ids)] == delim_ids:
                start_index = i + len(delim_ids)
                break

        if start_index == -1:
            logger.error("Agent delimiter not found in prompt: %s", full_prompt)
            labels = input_ids.copy()
        else:
            labels = [-100] * start_index + input_ids[start_index:]
            labels = labels[:len(input_ids)]

        example["input_ids"] = input_ids
        example["labels"] = labels
        return example

    dataset = dataset.map(preprocess_example)
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col not in ["input_ids", "labels"]])
    dataset.set_format("torch")
    return dataset

def build_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # 패딩 토큰이 없는 경우 추가
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    # "[AGENT]" special token이 없으면 추가
    if "[AGENT]" not in tokenizer.additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[AGENT]"]})
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    # 모델의 임베딩 크기를 새 tokenizer에 맞게 확장
    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Model vocab size: {model.config.vocab_size}, Tokenizer length: {len(tokenizer)}")
    # LoRA 구성 적용
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
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
