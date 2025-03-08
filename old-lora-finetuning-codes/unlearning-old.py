import torch
import numpy as np
import os
import json
from scipy.linalg import svd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

import numpy as np

class LoRAUnlearner:
    def __init__(self, alpha=0.5, beta=1.0, lambda_reg=0.1):
        self.alpha = alpha  # Adjustment coefficient for common weights
        self.beta = beta    # Unlearning strength hyperparameter
        self.lambda_reg = lambda_reg  # Regularization coefficient

    def svd_decompose(self, W):
        """Perform Singular Value Decomposition (SVD) on a weight matrix."""
        U, Sigma, Vt = np.linalg.svd(W, full_matrices=False)
        return U, np.diag(Sigma), Vt

    def extract_common_weights(self, W_plus, W_minus):
        """Extract common components between W+ and W- using shared subspace."""
        
        # Compute the common subspace basis
        W_cross = np.dot(W_plus, W_minus.T)
        U_c, _, _ = self.svd_decompose(W_cross)

        # Extract common weights
        W_common = np.dot(U_c, np.dot(U_c.T, (W_plus + W_minus) / 2))
        return W_common

    def remove_common_weights(self, W_minus, W_common):
        """Remove common components from W- using projection."""
        projection = np.dot(W_common, np.linalg.inv(np.dot(W_common.T, W_common)))
        projection = np.dot(projection, np.dot(W_common.T, W_minus))
        W_minus_pure = W_minus - projection
        return W_minus_pure

    def adjust_weights(self, W_minus_pure, fisher_matrix):
        """Adjust weights using Fisher information matrix."""
        fisher_inv = np.linalg.inv(fisher_matrix)
        W_minus_adjusted = np.multiply(fisher_inv, W_minus_pure - self.alpha * W_minus_pure)
        return W_minus_adjusted

    def unlearn_weights(self, W_plus, W_minus):
        """Perform the final unlearning process."""
        W_common = self.extract_common_weights(W_plus, W_minus)
        W_minus_pure = self.remove_common_weights(W_minus, W_common)
        

        # Final unlearning step
        W_unlearned = W_plus - self.beta * np.sign(np.multiply(W_plus, W_minus_pure)) * np.abs(W_minus_pure)
        W_unlearned = np.clip(W_unlearned, -1, 1)  # -1과 1 사이로 클리핑
        W_unlearned = W_unlearned - self.lambda_reg * W_unlearned
        W_unlearned = 0.9 * W_unlearned + 0.1 * W_plus


        return W_unlearned


#############################################
# 2. W_new를 LoRA의 두 행렬(lora_B, lora_A)로 분해하는 함수
#############################################
def factorize_weight(W_new: np.ndarray, r: int, scaling: float):
    """
    W_new(효과적인 LoRA 업데이트)를 lora_B와 lora_A로 분해합니다.
    LoRA 업데이트는 원래 (lora_B @ lora_A) * scaling 형태로 적용되므로,
    lora_B @ lora_A = W_new / scaling가 되어야 합니다.
    
    SVD를 통해 M = W_new/scaling = U S V^T 로 분해한 후,
    lora_B = U * sqrt(S)   (shape: [out_features, r])
    lora_A = sqrt(S) * V^T   (shape: [r, in_features])
    
    Args:
        W_new (np.ndarray): 결합된 effective weight update (out_features x in_features)
        r (int): LoRA의 rank (예제에서는 4)
        scaling (float): lora_alpha / r (예: 32/4 = 8)
        
    Returns:
        lora_B, lora_A: torch.Tensor로 변환된 분해 결과.
    """
    M = W_new / scaling  # (lora_B @ lora_A = M)
    U, S, Vh = np.linalg.svd(M, full_matrices=False)
    
    U_r = U[:, :r]      # (out_features x r)
    S_r = S[:r]         # (r,)
    Vh_r = Vh[:r, :]    # (r x in_features)
    
    sqrt_S = np.sqrt(S_r)
    lora_B = U_r * sqrt_S[np.newaxis, :]   # broadcasting, shape: (out_features x r)
    lora_A = sqrt_S[:, np.newaxis] * Vh_r    # shape: (r x in_features)
    
    # torch tensor로 변환 (dtype은 모델과 일치하도록)
    lora_B = torch.tensor(lora_B, dtype=torch.float16)
    lora_A = torch.tensor(lora_A, dtype=torch.float16)
    return lora_B, lora_A

#############################################
# 3. Qwen 모델 로드 및 LoRA 적용 (PEFT 방식)
#############################################
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
    #attn_implementation="eager"

)
model.config.sliding_window = None

# LoRA 설정 (원래 파인튜닝에 사용했던 target module 목록)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj", 
        "mlp.down_proj"
    ]
)

# PEFT를 통해 모델에 LoRA 어댑터 추가
peft_model = get_peft_model(model, lora_config)
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

path_plus = "./qwen-0.5b-lora-finetuned-alpaca-gpt4"   # W+
path_minus = "./qwen-0.5b-lora-finetuned-toxic"         # W-

# 원본 Qwen 모델 로드
model_name = "Qwen/Qwen2.5-0.5B"
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
)

# Fine-tuned LoRA 모델 불러오기
model_plus = PeftModel.from_pretrained(base_model, path_plus)
model_minus = PeftModel.from_pretrained(base_model, path_minus)

# 모델의 state_dict 가져오기
state_dict_plus = model_plus.state_dict()
state_dict_minus = model_minus.state_dict()
np.random.seed(42)
# 🟢 LoRA Target Modules
target_modules = [
    "self_attn.q_proj",
    "self_attn.k_proj", 
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj", 
    "mlp.down_proj"
]

# LoRA 설정 (새로운 모델에 적용)
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules
)

# 새로운 PEFT 모델 생성
new_peft_model = get_peft_model(base_model, lora_config)
new_state_dict = new_peft_model.state_dict()

# LoRA scaling factor
scaling = lora_config.lora_alpha / lora_config.r  # 예: 32/4 = 8

for layer_idx in range(24):  # Qwen-0.5B는 24개의 Transformer layer를 가짐
    for target_module in target_modules:
        # LoRA weight 키 생성 (각 레이어에 대해)
        key_A = f"base_model.model.model.layers.{layer_idx}.{target_module}.lora_A.default.weight"
        key_B = f"base_model.model.model.layers.{layer_idx}.{target_module}.lora_B.default.weight"

        # 키가 존재하는 경우에만 업데이트 수행
        if key_A in state_dict_plus and key_B in state_dict_plus:
            print(f"bringing W+ and W-{layer_idx}")
            # 기존 W+와 W- 불러오기 (torch.Tensor → numpy 변환)
            W_plus = (state_dict_plus[key_B] @ state_dict_plus[key_A]).cpu().numpy()
            W_minus = (state_dict_minus[key_B] @ state_dict_minus[key_A]).cpu().numpy()
            print("combining lora weights")
            # W_new 생성
            # rank_common < LoRA rank
            unlearner = LoRAUnlearner(alpha=0.5, beta=0.5, lambda_reg=0.1)
            W_new = unlearner.unlearn_weights(W_plus, W_minus)

            delta = np.linalg.norm(W_new - W_plus) / np.linalg.norm(W_plus)
            print(f"Layer {layer_idx} ΔW: {delta:.2%}")
            # W_new를 lora_A, lora_B로 복구 (함수 확인해야됨.)
            lora_B, lora_A = factorize_weight(W_new, r=lora_config.r, scaling=scaling)

            # 새로운 모델의 LoRA weight 업데이트 (torch.Tensor 형태로 변환) (음 진짜?)
            with torch.no_grad():
                new_state_dict[key_A].copy_(lora_A.to(new_state_dict[key_A].dtype))
                new_state_dict[key_B].copy_(lora_B.to(new_state_dict[key_B].dtype))
                
new_peft_model.load_state_dict(new_state_dict)



save_path = "./qwen-0.5b-unlearned-lora"
new_peft_model.save_pretrained(save_path)
print(f"새로운 결합된 LoRA 모델이 저장되었습니다: {save_path}")
