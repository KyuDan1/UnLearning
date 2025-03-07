import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import math
from tqdm import tqdm
from datasets import load_dataset
import random
import os

def calculate_perplexity(model, tokenizer, dataset, max_samples=100, max_length=512, device="cuda"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    # 샘플 수 제한
    if len(dataset) > max_samples:
        dataset = random.sample(list(dataset), max_samples)

    with torch.no_grad():
        for item in tqdm(dataset, desc="퍼플렉시티 계산 중"):
            # 텍스트 추출 (Hugging Face 데이터셋의 경우)
            text = item["text"] if isinstance(item, dict) and "text" in item else item

            # 텍스트 토크나이즈 (truncation, max_length 적용)
            encodings = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")

            # 토큰이 없는 경우 건너뛰기
            if encodings.input_ids.size(1) == 0:
                continue

            input_ids = encodings.input_ids.to(device).long()

            # 일부 모델은 attention_mask가 필요할 수 있음
            attention_mask = encodings.attention_mask.to(device) if "attention_mask" in encodings else None
            target_ids = input_ids.clone()

            if attention_mask is not None:
                outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            else:
                outputs = model(input_ids, labels=target_ids)

            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def load_evaluation_dataset(dataset_name="wikitext", subset="wikitext-2-raw-v1", split="test"):
    dataset = load_dataset(dataset_name, subset, split=split)
    return dataset

def evaluate_model(model_path, eval_dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")
    
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 베이스 모델 로드 후 LoRA 적용
    basemodel = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    basemodel.config.sliding_window = None  # 필요 시 추가 설정
    
    model = PeftModel.from_pretrained(basemodel, model_path)
    model.to(device)
    
    perplexity = calculate_perplexity(model, tokenizer, eval_dataset, max_samples=100, max_length=512, device=device)
    print(f"LoRA 모델 퍼플렉시티: {perplexity:.2f}")
    return perplexity

if __name__ == "__main__":
    # 평가 데이터셋을 한 번만 로드
    eval_dataset = load_evaluation_dataset()
    
    models = [f for f in os.listdir('.') if "qwen-" in f]
    with open('ppl.txt', 'w') as log:
        for model in models:
            print(f"\n모델 평가 중: {model}")
            perplexity = evaluate_model(model, eval_dataset)
            log.write(f"{model}: {perplexity:.2f}\n")
