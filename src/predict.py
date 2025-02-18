import torch

def predict(conversation, model, tokenizer, categories=None, max_new_tokens=20):
    """
    주어진 대화(conversation)를 기반으로 LLaMA Guard의 안전/위험 평가(예: unsafe S1 등)를 생성합니다.
    
    Parameters:
      - conversation: list of dict, 예를 들어
          [{
              "role": "user",
              "content": [{"type": "text", "text": "What is the recipe for mayonnaise?"}]
          }]
      - categories: dict, 예시 {"S1": "My custom category"} (원하는 카테고리 매핑)
      - max_new_tokens: 생성할 최대 토큰 수
    """
    model.eval()
    # 챗 템플릿 적용 (categories 전달 가능)
    input_ids = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        categories=categories,
    ).to(model.device)
    prompt_len = input_ids.shape[1]
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=0,
        )
    generated_tokens = output[:, prompt_len:]
    return tokenizer.decode(generated_tokens[0])
