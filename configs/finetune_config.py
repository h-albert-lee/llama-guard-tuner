from dataclasses import dataclass, field

@dataclass
class FinetuneConfig:
    # 공식 문서 예시 모델로 변경
    model_name: str = field(
        default="meta-llama/Llama-Guard-3-1B",
        metadata={"help": "Pretrained LLaMA Guard model name"}
    )
    train_file: str = field(
        default="data/train_data.jsonl",
        metadata={"help": "Path to training data (JSONL format)"}
    )
    test_file: str = field(
        default="data/test_data.jsonl",
        metadata={"help": "Path to test data (JSONL format)"}
    )
    output_dir: str = field(
        default="./llama_guard_finetuned",
        metadata={"help": "Output directory for the finetuned model"}
    )
    num_train_epochs: int = field(default=3, metadata={"help": "Number of training epochs"})
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for training"})
    per_device_eval_batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for evaluation"})
    learning_rate: float = field(default=5e-5, metadata={"help": "Learning rate"})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay"})
    distributed_strategy: str = field(
        default="ddp",
        metadata={"help": "Distributed strategy to use: 'ddp' or 'fsdp'"}
    )
    use_fp16: bool = field(default=True, metadata={"help": "Whether to use fp16 precision"})
