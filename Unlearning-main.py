import torch
import numpy as np
import os
import json
from scipy.linalg import svd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
import toxic_eval_generation
import toxic_eval
from dotenv import load_dotenv
import os
import huggingface_hub
os.environ["CUDA_VISIBLE_DEVICES"] = ""

load_dotenv()
token = os.environ.get('HUGGINGFACE_KEY')

huggingface_hub.login(token = token)

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
    num_layers = base_model_plus.config.num_hidden_layers

    for idx, module in enumerate(modules):
        target_module = module
        
        variance_diffs = []

        for layer_idx in range(num_layers):
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

import numpy as np

def deficiency_capability_unlearning(W_plus, W_minus, lambda_param):
    # 행 차원
    d = W_plus.shape[0]
    
    # 결과 저장할 리스트 초기화
    v_prime_list = []
    
    for i in range(d):
        # 각 행 벡터 가져오기
        v_plus = W_plus[i]
        v_minus = W_minus[i]
        
        # 벡터 정규화 (단위 벡터로 변환)
        v_plus_hat = v_plus / np.linalg.norm(v_plus)
        v_minus_hat = v_minus / np.linalg.norm(v_minus)
        
        # 일반 능력 방향 계산 (v_circ)
        v_circ = v_plus_hat + v_minus_hat
        
        # v_minus를 v_circ에 투영
        projection_scalar = np.dot(v_minus, v_circ) / np.dot(v_circ, v_circ)
        v_circ_minus = projection_scalar * v_circ
        
        # 결함 능력 추출 (Ext(v_minus))
        ext_v_minus = v_minus - v_circ_minus
        
        # 새 가중치 벡터 계산
        v_prime = v_plus - lambda_param * ext_v_minus
        
        # 결과 리스트에 추가
        v_prime_list.append(v_prime)
    
    # 모든 벡터를 쌓아서 새 행렬 생성
    W_prime = np.vstack(v_prime_list)
    
    return W_prime

def Unlearn(base_model,
            path_plus = "qwen-0.5b-lora-finetuned-alpacaPLUStoxic-0301",
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

            Ours = False,
            alpha = 1,
            save_path = "./qwen-0.5b-unlearned-lora-2025-0301",

            moving_alpha = False,
            alpha_start = 1,
            alpha_end = 3,

            var_alpha = False,
            dropout = 0.1,

            task_arithmetic = False,
            task_lambda = 0.2,
            
            Ext_Sub = False,
            Ext_Sub_lambda = 2.0,
                                ):
    
    


    model_name = base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model.config.sliding_window = None

    """    lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=rank*2,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=None
        )
    """

    #peft_model = get_peft_model(model, lora_config)
    path_plus = path_plus   # W+
    path_minus = path_minus         # W-

    # 원본 Qwen 모델 로드
    model_name = base_model
    base_model_plus = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, #device_map="auto"
    )
    base_model_minus = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, #device_map="auto"
    )

    # Fine-tuned LoRA 모델 불러오기
    model_plus = PeftModel.from_pretrained(base_model_plus, path_plus)
    model_minus = PeftModel.from_pretrained(base_model_minus, path_minus)

    state_dict_plus = model_plus.state_dict()
    state_dict_minus = model_minus.state_dict()

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=rank,
        lora_alpha=rank*2,
        lora_dropout=dropout,
        target_modules=target_modules
    )

    base_model_new = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16, #device_map="auto"
    )
    new_peft_model = get_peft_model(base_model_new, lora_config)
    new_state_dict = new_peft_model.state_dict()
    # LoRA scaling factor
    #scaling = lora_config.lora_alpha / lora_config.r  # 예: 32/4 = 8

    # variance difference alpha
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

    num_layers= model.config.num_hidden_layers
    for module in target_modules:
        
        for layer_idx in range(num_layers):
            key_A = f"base_model.model.model.layers.{layer_idx}.{module}.lora_A.default.weight"
            key_B = f"base_model.model.model.layers.{layer_idx}.{module}.lora_B.default.weight"

            W_plus = (state_dict_plus[key_B] @ state_dict_plus[key_A]).cpu().numpy()
            W_minus = (state_dict_minus[key_B] @ state_dict_minus[key_A]).cpu().numpy()

            # task_arithemtic
            if task_arithmetic == True:

                # 7번 Ablation Unlearning
                if moving_alpha == True:
                    d=(alpha_end - alpha_start)/(num_layers-1)
                    alpha = alpha_start + layer_idx*d
                    task_lambda = alpha
                new_W = W_plus - task_lambda * W_minus
                
            
            # Ext_Sub Method
            if Ext_Sub ==True:
                # 8번 Ablation Unlearning
                if moving_alpha == True:
                    d=(alpha_end - alpha_start)/(num_layers-1)
                    alpha = alpha_start + layer_idx*d
                    Ext_Sub_lambda = alpha

                new_W = deficiency_capability_unlearning(W_plus, W_minus, Ext_Sub_lambda)
                

            if Ours == True:
                
                U, S, Vt = svd(W_minus)
                
                U_toxic = U[:,:rank]
                U_proj = U_toxic@U_toxic.T

                toxic_of_Wplus = U_proj@W_plus
                
                if moving_alpha == True:
                    d=(alpha_end - alpha_start)/(num_layers-1)
                    alpha = alpha_start + layer_idx*d
                
                if var_alpha == True:
                    alpha = mapped_data[module][layer_idx]
                    print(f"alpha : {alpha}")

                new_W = W_plus - toxic_of_Wplus * alpha
                
            
            # 모든 방법론 공통 부분
            W_B, W_A = low_rank_decomposition(new_W,rank)
            W_B = torch.tensor(W_B, dtype=torch.float16)
            W_A = torch.tensor(W_A, dtype=torch.float16)
            new_state_dict[key_A].copy_(W_A.to(new_state_dict[key_A].dtype))
            new_state_dict[key_B].copy_(W_B.to(new_state_dict[key_B].dtype))
            print(f"{module}_{layer_idx}")

    new_peft_model.load_state_dict(new_state_dict)
    new_peft_model.save_pretrained(save_path)






if __name__ == "__main__":
    
    model_list = ["meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.3", "Qwen/Qwen2.5-7B"]
    dataset_list = [
            'alpaca_gpt4_data.json',
            'WizardLM_alpaca_evol_instruct_70k.json',
            'alpaca_gpt4_data_untruthful.json',
            'WizardLM_alpaca_evol_instruct_70k_untruthful.json',
            'toxic_train.json',
            'alpaca_plus_alpaca_untruthful.json',
            'WizardLM_plus_WizardLM_untruthful.json',
            'alpaca_plus_toxic.json',
            'WizardLM_plus_toxic.json',
        ]
    model_output_dir = []
    for model_source in model_list:
        for data_source in dataset_list:
            model_output_dir.append(f"outputModels/output_{model_source.replace('/','_')}_by_{data_source.replace('.json','')}")

    A = model_output_dir[0]
    B = model_output_dir[1]
    C = model_output_dir[2]
    D = model_output_dir[3]
    E = model_output_dir[4]
    F = model_output_dir[5]
    G = model_output_dir[6]
    H = model_output_dir[7]
    I = model_output_dir[8]




    models = ["meta-llama/Llama-3.1-8B", 
              #"mistralai/Mistral-7B-v0.3", 
              #"Qwen/Qwen2.5-7B"
              ]
    for base_model_name in models:
        
        
        bmn = base_model_name.replace('/', '_')
        """# 1. Task arithmetic
        #print("1. Task arithmetic")
        lambdas = [0.2, 0.2, 0.4, 0.2]
        save_paths = [f"{bmn}_task_arithmetic_1_a",
                      f"{bmn}_task_arithmetic_1_b",
                      f"{bmn}_task_arithmetic_1_c",
                      f"{bmn}_task_arithmetic_1_d",]
        models = [(A,C), (B,D), (A,E), (B,E)]
        for lambda_, save_path, model in zip(lambdas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    save_path=save_path,
                    task_arithmetic=True,
                    task_lambda=lambda_,

                    )
            print(f"{save_path}_unlearned")
"""
        # 2.Ext-Sub
        lambdas = [2.0, 2.0, 2.0, 2.0]
        save_paths = [f"{bmn}_Ext-Sub_2_a", 
                      f"{bmn}_Ext-Sub_2_b",
                      f"{bmn}_Ext-Sub_2_c",
                      f"{bmn}_Ext-Sub_2_d",]
        models = [(A,C), (B,D), (A,E), (B,E)]
        for lambda_, save_path, model in zip(lambdas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    save_path=save_path,
                    Ext_Sub=True,
                    Ext_Sub_lambda=lambda_,
                    
                    )
            print(f"{save_path}_unlearned")


        # 3. SVDP alpha constant
        alphas = [1, 1.5, 2, 2.5]
        save_paths = [f"{bmn}_SVDP_constant_3_a", 
                      f"{bmn}_SVDP_constant_3_b",
                      f"{bmn}_SVDP_constant_3_c",
                      f"{bmn}_SVDP_constant_3_d",]
        models = [(A,C), (B,D), (A,E), (B,E)]
        for lambda_, save_path, model in zip(alphas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    Ours=True,
                    save_path=save_path,
                    alpha = lambda_,
                                        )
            print(f"{save_path}_unlearned")

        # 4. SVDP alpha increasing
        alphas = [(1,2), (1,3), (2,3), (1.5,2.5)]
        save_paths = [f"{bmn}_SVDP_increasing_4_a", 
                      f"{bmn}_SVDP_increasing_4_b",
                      f"{bmn}_SVDP_increasing_4_c",
                      f"{bmn}_SVDP_increasing_4_d",]
        models = [(A,C), (B,D), (A,E), (B,E)]
        for lambda_, save_path, model in zip(alphas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    Ours=True,
                    save_path=save_path,
                    moving_alpha=True,
                    alpha_start=lambda_[0],
                    alpha_end=lambda_[1],
                                            )
            print(f"{save_path}_unlearned")
        

        # 5. SVDP alpha decreasing
        alphas = [(2,1), (3,1), (3,2), (2.5,1.5)]
        save_paths = [f"{bmn}_SVDP_decreasing_5_a", 
                      f"{bmn}_SVDP_decreasing_5_b",
                      f"{bmn}_SVDP_decreasing_5_c",
                      f"{bmn}_SVDP_decreasing_5_d",]
        models = [(A,C), (B,D), (A,E), (B,E)]
        for lambda_, save_path, model in zip(alphas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    Ours=True,
                    save_path=save_path,
                    moving_alpha=True,
                    alpha_start=lambda_[0],
                    alpha_end=lambda_[1],
                                            )
            print(f"{save_path}_unlearned")
            


        # 6. SVDP alpha layer-wise (BEST)
        alphas = [(2,1), (3,1), (3,2), (2.5,1.5)]
        save_paths = [f"{bmn}_SVDP_layerwise_6_a", 
                      f"{bmn}_SVDP_layerwise_6_b",
                      f"{bmn}_SVDP_layerwise_6_c",
                      f"{bmn}_SVDP_layerwise_6_d",]
        models = [(A,C), (B,D), (A,E), (B,E)]
        for lambda_, save_path, model in zip(alphas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    Ours=True,
                    save_path=save_path,
                    var_alpha=True,
                    alpha_start=lambda_[0],
                    alpha_end=lambda_[1],
                                            )
            print(f"{save_path}_unlearned")
            

        # 7. Ablation Unlearning (단순 task arithmetic)
        alphas = [(2,1), (3,1), (3,2), (2.5,1.5)] # task arithmetic의 lambda이므로 좀 달라져야 함.
        save_paths = [f"{bmn}_Ablation_taskarithmetic_7_a", 
                      f"{bmn}_Ablation_taskarithmetic_7_b",
                      f"{bmn}_Ablation_taskarithmetic_7_c",
                      f"{bmn}_Ablation_taskarithmetic_7_d",]
        models = [(A,C), (B,D), (A,E), (B,E)]
        for lambda_, save_path, model in zip(alphas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    task_arithmetic=True,
                    save_path=save_path,
                    moving_alpha=True,
                    alpha_start=lambda_[0],
                    alpha_end=lambda_[1],
                                            )
            print(f"{save_path}_unlearned")
            
        # 8. Ablation Unlearning (Ext-Sub)
        alphas = [(2,1), (3,1), (3,2), (2.5,1.5)] # task Ext-Sub의 lambda이므로 좀 달라져야 함.
        save_paths = [f"{bmn}_Ablation_Ext_Sub_8_a", 
                      f"{bmn}_Ablation_Ext_Sub_8_b",
                      f"{bmn}_Ablation_Ext_Sub_8_c",
                      f"{bmn}_Ablation_Ext_Sub_8_d",]
        models = [(A,C), (B,D), (A,E), (B,E)]
        for lambda_, save_path, model in zip(alphas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    Ext_Sub=True,
                    save_path=save_path,
                    moving_alpha=True,
                    alpha_start=lambda_[0],
                    alpha_end=lambda_[1],
                                            )
            print(f"{save_path}_unlearned")


        # 9. Bootstrap Unlearning (layerwise)
        alphas = [(2,1), (3,1), (3,2), (2.5,1.5)]
        save_paths = [f"{bmn}_Bootstrap_layerwise_9_a", 
                      f"{bmn}_Bootstrap_layerwise_9_b",
                      f"{bmn}_Bootstrap_layerwise_9_c",
                      f"{bmn}_Bootstrap_layerwise_9_d",]
        models = [(F,C), (G,D), (H,E), (I,E)]
        for lambda_, save_path, model in zip(alphas, save_paths, models):
            print(f"{save_path}_unlearning")
            Unlearn(base_model=base_model_name,
                    path_plus=model[0],
                    path_minus=model[1],
                    rank = 16,
                    Ours=True,
                    save_path=save_path,
                    var_alpha=True,
                    alpha_start=lambda_[0],
                    alpha_end=lambda_[1],
                                            )
            print(f"{save_path}_unlearned")