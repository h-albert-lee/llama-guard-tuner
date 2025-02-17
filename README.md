# llama-guard-tuner
Let's finetune llama-guard

This repository provides an example implementation for fine-tuning the LLaMA Guard model using PEFT (LoRA) and multi-GPU distributed training with Hugging Face Accelerate. The code is modularized into several components for easy maintenance and extension.

# Usage

## 1. Install Packages
```bash
pip install -r requirements.txt
```

## 2. Data Preparation
Prepare your data by placing the `train_data.jsonl` and `test_data.jsonl` files in the `data/` directory.

## 3. Execute Training
Run the following command to start the fine-tuning process using Hugging Face Accelerate:
```bash
accelerate launch scripts/finetune.py --config_file accelerate_config.yaml
```

## 4. Inference
After training, you can use the `src/predict.py` module to perform model predictions.
