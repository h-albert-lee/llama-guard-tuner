#!/usr/bin/env python
import logging
from configs.finetune_config import FinetuneConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from src.prompt_builder import build_training_prompt, AgentType, ConversationTurn, load_safety_categories

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(train_file: str, test_file: str, tokenizer):
    """
    학습 데이터를 불러오고, 각 예제에 대해 전체 시스템 프롬프트(안전 정책 + 카테고리 목록)와
    사용자 대화(assistant 응답 포함)를 하나의 입력 텍스트로 구성
    """
    dataset = load_dataset("json", data_files={"train": train_file, "test": test_file})
    
    def preprocess_example(example):
        try:
            user_text = example["conversation"][0]["content"][0]["text"]
            assistant_text = example["conversation"][1]["content"][0]["text"]
        except (KeyError, IndexError):
            logger.error("Malformed conversation: %s", example)
            return example
        
        conv_turns = [
            ConversationTurn(message=user_text, agent_type=AgentType.USER),
            ConversationTurn(message=assistant_text, agent_type=AgentType.AGENT)
        ]
        safety_categories = load_safety_categories()
        full_prompt = build_training_prompt(
            agent_type=AgentType.AGENT,
            conversations=conv_turns,
            categories=safety_categories,
            category_short_name_prefix="S",
            with_policy=True
        )
        example["input_ids"] = tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)["input_ids"]
        return example

    dataset = dataset.map(preprocess_example)
    dataset = dataset.remove_columns([col for col in dataset["train"].column_names if col != "input_ids"])
    dataset.set_format("torch")
    return dataset

def build_model(config: FinetuneConfig):
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

def get_training_args(config: FinetuneConfig):
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
