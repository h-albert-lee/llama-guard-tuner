#!/usr/bin/env python
import logging
from configs.finetune_config import FinetuneConfig
from src.data_utils import load_and_preprocess_data
from src.model_utils import build_model
from src.trainer_utils import get_training_arguments, train_model
from src.predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. 설정 불러오기
    config = FinetuneConfig()

    # 2. 모델 및 토크나이저 로드 (LoRA 적용)
    model, tokenizer = build_model(config)

    # 3. 데이터 로드 및 전처리
    dataset = load_and_preprocess_data(config.train_file, config.test_file, tokenizer)

    # 4. TrainingArguments 설정 (멀티-GPU 분산 학습 포함)
    training_args = get_training_arguments(config)

    # 5. 모델 학습 및 평가
    trainer, train_result, metrics = train_model(model, tokenizer, dataset, training_args)

    # 6. 간단한 추론 테스트
    test_text = "This is a harmful statement."
    pred = predict(test_text, model, tokenizer)
    logger.info(f"Test Text: {test_text}\nPrediction: {pred}")

if __name__ == "__main__":
    main()
