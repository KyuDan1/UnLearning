import torch
import numpy as np
import os
import json
from scipy.linalg import svd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import toxic_eval_generation
import toxic_eval
def low_rank_decomposition(matrix, rank):
    # SVD를 사용하여 행렬 분해
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    
    # 상위 r개의 특이값만 사용
    U_r = U[:, :rank]
    S_r = np.diag(S[:rank])
    Vt_r = Vt[:rank, :]
    
    # W_A와 W_B 계산
    W_A = U_r @ np.sqrt(S_r)
    W_B = np.sqrt(S_r) @ Vt_r
    
    return W_A, W_B

def svd(W):
        U, Sigma, Vt = np.linalg.svd(W, full_matrices=False)
        return U, np.diag(Sigma), Vt
    
def map_alpha(diff_list, a_min=1, a_max=2):
    min_val = min(diff_list)
    max_val = max(diff_list)
    
    if max_val == min_val:
        return [a_max for _ in diff_list]
    
    alphas = []
    for x in diff_list:
        # x가 min_val일 때 0, max_val일 때 1이 되도록 정규화
        norm = (x - min_val) / (max_val - min_val)
        alpha = a_max - norm * (a_max - a_min)
        alphas.append(alpha)

    return alphas

def get_variance_diffs(path_plus="qwen-0.5b-lora-finetuned-alpacaPLUStoxic-0301",
                        path_minus="qwen-0.5b-lora-finetuned-toxic" ,
                        modules=[
                                    "self_attn.q_proj",
                                    "self_attn.k_proj", 
                                    "self_attn.v_proj",
                                    "self_attn.o_proj",
                                    "mlp.gate_proj",
                                    "mlp.up_proj", 
                                    "mlp.down_proj"
                                ]):
            # 원본 Qwen 모델 로드
    model_name = "Qwen/Qwen2.5-0.5B"
    base_model_plus = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )
    base_model_minus = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )

    # Fine-tuned LoRA 모델 불러오기
    model_plus = PeftModel.from_pretrained(base_model_plus, path_plus)
    model_minus = PeftModel.from_pretrained(base_model_minus, path_minus)
    state_dict_plus = model_plus.state_dict()
    state_dict_minus = model_minus.state_dict()
    print("getting variance differences")
    modules_var = {}
    for idx, module in enumerate(modules):
        target_module = module
        
        variance_diffs = []

        for layer_idx in range(24):
            key_A = f"base_model.model.model.layers.{layer_idx}.{target_module}.lora_A.default.weight"
            key_B = f"base_model.model.model.layers.{layer_idx}.{target_module}.lora_B.default.weight"
            
            W_plus = (state_dict_plus[key_B] @ state_dict_plus[key_A]).cpu().numpy()
            W_minus = (state_dict_minus[key_B] @ state_dict_minus[key_A]).cpu().numpy()
            
            # 각 W의 분산 계산
            var_plus = np.var(W_plus)
            var_minus = np.var(W_minus)
            
            # 분산의 차이 계산
            variance_diff = var_minus - var_plus
            variance_diffs.append(variance_diff)
        
        modules_var[module]=variance_diffs
    return modules_var


def Unlearn(path_plus = "qwen-0.5b-lora-finetuned-alpacaPLUStoxic-0301",
            path_minus = "qwen-0.5b-lora-finetuned-toxic",
                target_modules = [
                "self_attn.q_proj",
                "self_attn.k_proj", 
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj", 
                "mlp.down_proj"
            ],
            rank = 4,
            alpha = 1,
            save_path = "./qwen-0.5b-unlearned-lora-2025-0301",
            moving_alpha = False,
            alpha_start = 1,
            alpha_end = 3,
            var_alpha = False,
                                ):
    
    


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
        target_modules=target_modules
    )

    #peft_model = get_peft_model(model, lora_config)
    path_plus = path_plus   # W+
    path_minus = path_minus         # W-

    # 원본 Qwen 모델 로드
    model_name = "Qwen/Qwen2.5-0.5B"
    base_model_plus = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )
    base_model_minus = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )

    # Fine-tuned LoRA 모델 불러오기
    model_plus = PeftModel.from_pretrained(base_model_plus, path_plus)
    model_minus = PeftModel.from_pretrained(base_model_minus, path_minus)

    state_dict_plus = model_plus.state_dict()
    state_dict_minus = model_minus.state_dict()

    target_modules = target_modules

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )

    base_model_new = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
    )
    new_peft_model = get_peft_model(base_model_new, lora_config)
    new_state_dict = new_peft_model.state_dict()
    # LoRA scaling factor
    scaling = lora_config.lora_alpha / lora_config.r  # 예: 32/4 = 8


    if var_alpha == True:
        dict = get_variance_diffs(path_plus="qwen-0.5b-lora-finetuned-alpacaPLUStoxic-0301",
                        path_minus="qwen-0.5b-lora-finetuned-toxic" ,
                        modules=[
                                    "self_attn.q_proj",
                                    "self_attn.k_proj", 
                                    "self_attn.v_proj",
                                    "self_attn.o_proj",
                                    "mlp.gate_proj",
                                    "mlp.up_proj", 
                                    "mlp.down_proj"
                                ])
        
        mapped_data = {key: map_alpha(value, min(alpha_start, alpha_end), max(alpha_start, alpha_end)) for key, value in dict.items()}


    for module in target_modules:
        target_module = module

        for layer_idx in range(24):
            key_A = f"base_model.model.model.layers.{layer_idx}.{target_module}.lora_A.default.weight"
            key_B = f"base_model.model.model.layers.{layer_idx}.{target_module}.lora_B.default.weight"

            W_plus = (state_dict_plus[key_B] @ state_dict_plus[key_A]).cpu().numpy()
            W_minus = (state_dict_minus[key_B] @ state_dict_minus[key_A]).cpu().numpy()



            U, S, Vt = svd(W_minus)
            
            U_toxic = U[:,:rank]
            U_proj = U_toxic@U_toxic.T

            toxic_of_Wplus = U_proj@W_plus
            
            if moving_alpha == True:
                d=(alpha_end - alpha_start)/(24-1)
                alpha = alpha_start + layer_idx*d
            
            if var_alpha == True:
                alpha = mapped_data[module][layer_idx]
                print(f"alpha : {alpha}")

            new_W = W_plus - toxic_of_Wplus * alpha
            W_B, W_A = low_rank_decomposition(new_W,rank)
            W_B = torch.tensor(W_B, dtype=torch.float16)
            W_A = torch.tensor(W_A, dtype=torch.float16)
            new_state_dict[key_A].copy_(W_A.to(new_state_dict[key_A].dtype))
            new_state_dict[key_B].copy_(W_B.to(new_state_dict[key_B].dtype))
            print(f"{module}_{layer_idx}")

    new_peft_model.load_state_dict(new_state_dict)
    save_path = save_path
    new_peft_model.save_pretrained(save_path)


# static alpha experiments
"""alphas = [1.5,2.5]
# 이름에 qwen-lora가 들어가야됨..
save_paths = ["qwen-lora-unlearned-alpha-1-5", "qwen-lora-unlearned-alpha-2-5"]

for alpha, save_path in zip(alphas, save_paths):
    Unlearn(alpha = alpha, save_path = save_path)
    output_path = "small_output/"+save_path+".json"
    toxic_eval_generation.main(model_name_or_path=save_path,
                               batch_size=1,
                               input_path="data/toxic_test_small.json",
                               output_path=output_path)
    toxic_eval.main(input_path=output_path,
                    output_path=output_path.split('.json')[0]+"result.json",
                    checkpoint_path="checkpoints/toxic_debiased-c7548aa0.ckpt")"""


# linearly increasing alpha experiments
"""alphas = [(1,2),(1,3),(2,3), (1.5,2.5)]

# 이름에 qwen-lora가 들어가야됨..
save_paths = ["qwen-lora-unlearned-movingalpha-1-2", "qwen-lora-unlearned-movingalpha-1-3",
              "qwen-lora-unlearned-movingalpha-2-3", "qwen-lora-unlearned-movingalpha-15-25"]

for alpha, save_path in zip(alphas, save_paths):
    Unlearn(save_path = save_path, moving_alpha=True, alpha_start=alpha[0], alpha_end=alpha[1])
    output_path = "small_output/"+save_path+".json"
    toxic_eval_generation.main(model_name_or_path=save_path,
                               batch_size=1,
                               input_path="data/toxic_test_small.json",
                               output_path=output_path)
    toxic_eval.main(input_path=output_path,
                    output_path=output_path.split('.json')[0]+"result.json",
                    checkpoint_path="checkpoints/toxic_debiased-c7548aa0.ckpt")"""
    
# linearly decreasing alpha experiments

"""alphas = [(2, 1),(3,1),(3,2), (2.5,1.5)]
# 이름에 qwen-lora가 들어가야됨..
save_paths = ["qwen-lora-unlearned-movingalpha-2-1", "qwen-lora-unlearned-movingalpha-3-1",
              "qwen-lora-unlearned-movingalpha-3-2", "qwen-lora-unlearned-movingalpha-25-15"]

for alpha, save_path in zip(alphas, save_paths):
    Unlearn(save_path = save_path, moving_alpha=True, alpha_start=alpha[0], alpha_end=alpha[1])
    output_path = "small_output/"+save_path+".json"
    toxic_eval_generation.main(model_name_or_path=save_path,
                               batch_size=1,
                               input_path="data/toxic_test_small.json",
                               output_path=output_path)
    toxic_eval.main(input_path=output_path,
                    output_path=output_path.split('.json')[0]+"result.json",
                    checkpoint_path="checkpoints/toxic_debiased-c7548aa0.ckpt")"""

# dev diff based alpha experiments
alphas = [(2, 1),(3,1),(3,2), (2.5,1.5)]
# 이름에 qwen-lora가 들어가야됨..
save_paths = ["qwen-lora-unlearned-devalpha-2-1", "qwen-lora-unlearned-devalpha-3-1",
              "qwen-lora-unlearned-devalpha-3-2", "qwen-lora-unlearned-devalpha-25-15"]

for alpha, save_path in zip(alphas, save_paths):
    Unlearn(save_path = save_path, alpha_start=alpha[0], alpha_end=alpha[1], var_alpha=True)
    output_path = "small_output/"+save_path+".json"
    toxic_eval_generation.main(model_name_or_path=save_path,
                               batch_size=1,
                               input_path="data/toxic_test_small.json",
                               output_path=output_path)
    toxic_eval.main(input_path=output_path,
                    output_path=output_path.split('.json')[0]+"result.json",
                    checkpoint_path="checkpoints/toxic_debiased-c7548aa0.ckpt")