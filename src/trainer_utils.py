import os
import logging
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)

def get_training_arguments(config):
    """
    분산 훈련(DDP/FSDP) 설정을 포함한 TrainingArguments를 생성합니다.
    """
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
        logging_dir=os.path.join(config.output_dir, "logs"),
        logging_steps=50,
        ddp_find_unused_parameters=False,
    )
    if config.distributed_strategy.lower() == "fsdp":
        logger.info("Using FSDP for distributed training.")
        training_args.fsdp = "auto"
    return training_args

def train_model(model, tokenizer, dataset, training_args):
    """
    Trainer를 이용해 LM fine-tuning을 진행합니다.
    DataCollatorForLanguageModeling을 사용해 causal LM 학습 목표에 맞게 데이터를 준비합니다.
    """
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model()  # 최종 체크포인트 저장

    metrics = trainer.evaluate()
    logger.info(f"Evaluation metrics: {metrics}")

    tokenizer.save_pretrained(training_args.output_dir)
    return trainer, train_result, metrics
