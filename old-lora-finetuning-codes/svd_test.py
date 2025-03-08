import torch
import torch.nn as nn
from peft import PeftModel
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)

def get_lora_weights(model):
    """
    LoRA 가중치를 추출하는 함수
    Returns: Dictionary of layer names and their corresponding 'A'/'B' matrices
    """
    lora_weights = {}
    
    # named_modules() 대신 named_parameters()를 사용
    for name, param in model.named_parameters():
        # 파라미터 이름에 lora_A나 lora_B가 들어있는지 확인
        if 'lora_A' in name or 'lora_B' in name:
            # weight 이름에서 base_name 추출
            # ex) transformer.blocks.0.attention.lora_A.weight -> transformer.blocks.0.attention
            base_name = name.replace('.lora_A.weight', '').replace('.lora_B.weight', '')
            if base_name not in lora_weights:
                lora_weights[base_name] = {}

            # 실제 파라미터값을 CPU 텐서로 복사
            if 'lora_A' in name:
                lora_weights[base_name]['A'] = param.detach().cpu()
            else:
                lora_weights[base_name]['B'] = param.detach().cpu()

    return lora_weights

def compute_delta_w(model, lora_weights):
    """
    각 레이어의 Delta W (B*A) 계산
    model의 LoRA 설정(스케일)도 함께 고려
    """
    delta_w = {}
    
    # PEFT LoRA Config 가져오기 (여기서는 'default'라고 가정)
    lora_config = model.peft_config['default']
    scale = lora_config.lora_alpha / lora_config.r

    for layer_name, weights in lora_weights.items():
        # A, B가 모두 있어야 Delta W를 계산
        if 'A' in weights and 'B' in weights:
            delta_w[layer_name] = torch.matmul(weights['B'], weights['A']) * scale
    
    return delta_w

def analyze_svd(delta_w):
    """
    각 Delta W에 대해 SVD 분석 수행
    """
    svd_results = {}
    for layer_name, weight in delta_w.items():
        weight_np = weight.numpy()
        U, S, Vt = svd(weight_np, full_matrices=False)
        
        svd_results[layer_name] = {
            'singular_values': S,
            'U': U,
            'Vt': Vt,
            'rank': np.sum(S > 1e-10)  # 유효 rank
        }
    
    return svd_results

def plot_singular_values(svd_results, save_path="singular_values.png"):
    """
    특이값 분포 시각화
    """
    plt.figure(figsize=(15, 10))
    for layer_name, result in svd_results.items():
        S = result['singular_values']
        plt.semilogy(range(1, len(S) + 1), S, label=layer_name)
    
    plt.xlabel('Index')
    plt.ylabel('Singular Value (log scale)')
    plt.title('Singular Value Distribution of LoRA Delta W')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 학습된 LoRA 모델 로드
    base_model_path = "Qwen/Qwen2.5-0.5B"
    lora_model_path = "./qwen-0.5b-lora-finetuned"
    
    # Base 모델
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # LoRA 모델
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path
    )
    
    # LoRA 가중치 추출
    lora_weights = get_lora_weights(model)
    
    # Delta W 계산
    delta_w = compute_delta_w(model, lora_weights)
    
    # SVD 분석
    svd_results = analyze_svd(delta_w)
    
    # 특이값 플롯
    plot_singular_values(svd_results)
    
    # 각 레이어의 rank, 일부 특이값 정보 출력
    for layer_name, result in svd_results.items():
        singular_values = result['singular_values']
        print(f"\nLayer: {layer_name}")
        print(f"Effective rank: {result['rank']}")
        print(f"Top 5 singular values: {singular_values[:5]}")
        
        # 조건수
        cond_number = (singular_values[0] / singular_values[-1] 
                       if singular_values[-1] != 0 else float('inf'))
        print(f"Condition number: {cond_number}")

if __name__ == "__main__":
    main()
