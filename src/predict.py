import torch

def predict(text: str, model, tokenizer):
    """
    주어진 문장에 대해 안전/위험 여부를 예측합니다.
    """
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "safe" if prediction == 1 else "unsafe"
