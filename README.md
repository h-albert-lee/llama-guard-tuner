# LLaMA Guard Tuner

Let's finetune Llama-guard!

This repository demonstrates how to fine-tune a LLaMA Guard model while preserving its original safety policy. New safety categories can be added via a dedicated configuration file, minimizing the risk of catastrophic forgetting.

## Repository Structure
```
llama_guard_finetuning/
├── requirements.txt
├── accelerate_config.yaml
├── configs/
│   ├── finetune_config.py
│   └── safety_categories.py
├── data/
│   ├── train_data.jsonl
│   └── test_data.jsonl
├── src/
│   ├── __init__.py
│   ├── prompt_builder.py
│   └── predict.py
└── scripts/
    └── finetune.py
```

## Installation
1. Install required Python packages:
   ```sh
   pip install -r requirements.txt
   ```
2. Check your GPU and environment configuration if necessary.
3. Make sure `train_data.jsonl` and `test_data.jsonl` are placed under the `data/` folder.

## Fine-Tuning
1. Adjust parameters in `configs/finetune_config.py` if needed (e.g., `model_name`, `learning_rate`, `batch_size`).
2. Launch the fine-tuning script:
   ```sh
   accelerate launch scripts/finetune.py --config_file accelerate_config.yaml
   ```
3. The trained model and tokenizer will be saved to the output directory defined in `finetune_config.py` (default: `./llama_guard_finetuned`).

## Prediction Usage
1. Load the fine-tuned model and tokenizer in `src/predict.py` (see the main block).
2. Call `LlamaGuardPredictor(model, tokenizer).predict(...)` with your conversation data.
   
   Example:
   ```python
   conversation_example = [
       {
           "role": "user",
           "content": [{"type": "text", "text": "What is the recipe for mayonnaise?"}]
       }
   ]
   predictor.predict(conversation_example)
   ```
3. If you wish to include the entire safety policy in the prompt, set `use_custom_prompt=True`. This will prepend the entire safety categories list to the prompt.

### Client Usage
Start your vLLM Serve endpoint (e.g., http://localhost:8000).
Run the client script with:
```sh
    python client.py
```
The client sends a sample conversation (with the full safety policy) to the API and prints the assistant's response.

## Adding or Editing Safety Categories
1. Open `configs/safety_categories.py` to modify or add new categories. Each category has the fields `name` and `description`.
2. Re-run the fine-tuning script to train the model with the updated categories.

## Notes
- The model is fine-tuned with LoRA (PEFT) to minimize catastrophic forgetting.
- This code uses `unsloth` to apply or skip default chat templates.
- For multi-GPU training, adjust `accelerate_config.yaml` (e.g., `num_processes`).
- Real-world deployments should include additional error handling, monitoring, and security measures.