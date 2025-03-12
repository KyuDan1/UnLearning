import torch
import numpy as np
import os
import json
from transformers import AutoTokenizer, AutoConfig
from peft import LoraConfig
from safetensors.torch import load_file, save_file
import time
from tqdm import tqdm
import gc

# GPU 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def low_rank_decomposition_gpu(matrix, rank):
    """GPU 가속 저차원 분해"""
    # PyTorch SVD를 사용하여 행렬 분해
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    
    # 상위 r개의 특이값만 사용
    U_r = U[:, :rank]
    S_r = torch.diag(S[:rank])
    Vt_r = Vt[:rank, :]
    
    # W_A와 W_B 계산
    W_A = U_r @ torch.sqrt(S_r)
    W_B = torch.sqrt(S_r) @ Vt_r
    
    return W_B, W_A

def svd_gpu(W):
    """GPU 가속 SVD"""
    U, Sigma, Vt = torch.linalg.svd(W, full_matrices=False)
    return U, torch.diag(Sigma), Vt

def map_alpha(diff_list, a_min=1, a_max=2):
    """알파 값 매핑"""
    min_val = min(diff_list)
    max_val = max(diff_list)
    
    if max_val == min_val:
        return [a_max for _ in diff_list]
    
    alphas = []
    for x in diff_list:
        norm = (x - min_val) / (max_val - min_val)
        alpha = a_max - norm * (a_max - a_min)
        alphas.append(alpha)

    return alphas

def extract_layer_module_info(key):
    """키에서 레이어 번호와 모듈 이름을 추출합니다."""
    parts = key.split('.')
    
    # 레이어 번호 찾기
    layer_idx = None
    for i, part in enumerate(parts):
        if part == "layers" and i+1 < len(parts) and parts[i+1].isdigit():
            layer_idx = int(parts[i+1])
            break
    
    # 모듈 이름 찾기
    module_name = None
    module_keywords = ["self_attn", "mlp"]
    
    for i, part in enumerate(parts):
        if part in module_keywords and i+1 < len(parts):
            if parts[i+1] in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                module_name = f"{part}.{parts[i+1]}"
                break
    
    return layer_idx, module_name

def is_lora_A_key(key):
    """키가 LoRA A 행렬에 해당하는지 확인합니다."""
    return "lora_A" in key and "weight" in key

def is_lora_B_key(key):
    """키가 LoRA B 행렬에 해당하는지 확인합니다."""
    return "lora_B" in key and "weight" in key

def find_matching_B_key(A_key, all_keys):
    """주어진 A 키에 대응하는 B 키를 찾습니다."""
    B_key = A_key.replace("lora_A", "lora_B")
    if B_key in all_keys:
        return B_key
    
    parts = A_key.split('.')
    for i, part in enumerate(parts):
        if "lora_A" in part:
            parts[i] = part.replace("lora_A", "lora_B")
            potential_B_key = '.'.join(parts)
            if potential_B_key in all_keys:
                return potential_B_key
    
    return None

def deficiency_capability_unlearning_gpu(W_plus, W_minus, lambda_param):
    """GPU 가속 결함 능력 언러닝"""
    # 행 차원
    d = W_plus.shape[0]
    
    # 결과 텐서 초기화
    v_prime = torch.zeros_like(W_plus)
    
    for i in range(d):
        # 각 행 벡터 가져오기
        v_plus = W_plus[i]
        v_minus = W_minus[i]
        
        # 벡터 정규화 (단위 벡터로 변환)
        v_plus_norm = torch.norm(v_plus)
        v_minus_norm = torch.norm(v_minus)
        
        # 0으로 나누기 방지
        if v_plus_norm == 0 or v_minus_norm == 0:
            v_prime[i] = v_plus
            continue
            
        v_plus_hat = v_plus / v_plus_norm
        v_minus_hat = v_minus / v_minus_norm
        
        # 일반 능력 방향 계산 (v_circ)
        v_circ = v_plus_hat + v_minus_hat
        
        # v_circ가 0벡터인지 확인
        v_circ_norm = torch.norm(v_circ)
        if v_circ_norm == 0:
            v_prime[i] = v_plus
            continue
            
        # v_minus를 v_circ에 투영
        projection_scalar = torch.dot(v_minus, v_circ) / torch.dot(v_circ, v_circ)
        v_circ_minus = projection_scalar * v_circ
        
        # 결함 능력 추출 (Ext(v_minus))
        ext_v_minus = v_minus - v_circ_minus
        
        # 새 가중치 벡터 계산
        v_prime[i] = v_plus - lambda_param * ext_v_minus
    
    return v_prime

def get_variance_diffs_efficient(path_plus, path_minus, target_modules, num_layers=None, debug=False):
    """분산 차이를 효율적으로 계산하는 함수 - 어댑터 가중치만 사용"""
    print("Calculating variance differences...")
    
    # 어댑터 가중치 로드
    plus_state_dict = load_file(f"{path_plus}/adapter_model.safetensors", device="cpu")
    minus_state_dict = load_file(f"{path_minus}/adapter_model.safetensors", device="cpu")
    
    # 설정 로드
    with open(f"{path_plus}/adapter_config.json", 'r') as f:
        adapter_config = json.load(f)
    
    # num_layers가 없는 경우 추정
    if num_layers is None:
        # 레이어 수 추정을 위해 키 분석
        layer_indices = set()
        for key in plus_state_dict.keys():
            if is_lora_A_key(key):
                layer_idx, _ = extract_layer_module_info(key)
                if layer_idx is not None:
                    layer_indices.add(layer_idx)
        
        num_layers = max(layer_indices) + 1 if layer_indices else 32
        if debug:
            print(f"Estimated number of layers: {num_layers}")
    
    # A 키 목록 생성
    A_keys_plus = [k for k in plus_state_dict.keys() if is_lora_A_key(k)]
    
    # 결과 저장용 딕셔너리
    modules_var = {module: [] for module in target_modules}
    
    # 레이어와 모듈별로 키 그룹화
    layer_module_keys = {}
    for A_key in A_keys_plus:
        layer_idx, module_name = extract_layer_module_info(A_key)
        if layer_idx is not None and module_name is not None:
            layer_module_keys[(layer_idx, module_name)] = A_key
    
    # 각 모듈 및 레이어에 대해 분산 차이 계산
    for module in target_modules:
        variance_diffs = []
        
        for layer_idx in range(num_layers):
            if (layer_idx, module) not in layer_module_keys:
                # 해당 레이어/모듈 조합이 없으면 기본값 사용
                variance_diffs.append(0.0)
                continue
                
            A_key = layer_module_keys[(layer_idx, module)]
            B_key = find_matching_B_key(A_key, plus_state_dict.keys())
            
            if B_key is None or A_key not in minus_state_dict or B_key not in minus_state_dict:
                variance_diffs.append(0.0)
                continue
                
            # GPU로 이동하여 계산
            A_plus = plus_state_dict[A_key].to(DEVICE)
            B_plus = plus_state_dict[B_key].to(DEVICE)
            A_minus = minus_state_dict[A_key].to(DEVICE)
            B_minus = minus_state_dict[B_key].to(DEVICE)
            
            W_plus = B_plus @ A_plus
            W_minus = B_minus @ A_minus
            
            # 각 W의 분산 계산
            var_plus = torch.var(W_plus).item()
            var_minus = torch.var(W_minus).item()
            
            # 분산의 차이 계산
            variance_diff = var_minus - var_plus
            variance_diffs.append(variance_diff)
            
            # GPU 메모리 정리
            del A_plus, B_plus, A_minus, B_minus, W_plus, W_minus
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        modules_var[module] = variance_diffs
    
    return modules_var

def Unlearn(base_model,
            path_plus,
            path_minus,
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
            save_path = "./unlearned-lora",
            moving_alpha = False,
            alpha_start = 1,
            alpha_end = 3,
            var_alpha = False,
            dropout = 0.1,
            task_arithmetic = False,
            task_lambda = 0.2,
            Ext_Sub = False,
            Ext_Sub_lambda = 2.0,
            debug = False,
            batch_size = 5,
            ):
    
    start_time = time.time()
    
    print(f"Unlearning {path_plus} - {path_minus} with method: {('SVDP' if Ours else 'Task Arithmetic' if task_arithmetic else 'Ext-Sub' if Ext_Sub else 'Unknown')}")
    
    # 모델 설정 로드 - 토크나이저는 필요한 경우에만 로드
    try:
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        num_layers = config.num_hidden_layers
        print(f"Model has {num_layers} layers")
    except Exception as e:
        print(f"Error loading config: {e}, estimating layers from adapter weights")
        num_layers = None  # 나중에 어댑터 키에서 추정
    
    # 어댑터 가중치 로드
    print(f"Loading adapter weights...")
    plus_state_dict = load_file(f"{path_plus}/adapter_model.safetensors", device="cpu")
    minus_state_dict = load_file(f"{path_minus}/adapter_model.safetensors", device="cpu")
    
    # 어댑터 설정 로드 및 복사
    with open(f"{path_plus}/adapter_config.json", 'r') as f:
        adapter_config = json.load(f)
    
    # num_layers가 없으면 추정
    if num_layers is None:
        layer_indices = set()
        for key in plus_state_dict.keys():
            if is_lora_A_key(key):
                layer_idx, _ = extract_layer_module_info(key)
                if layer_idx is not None:
                    layer_indices.add(layer_idx)
        
        num_layers = max(layer_indices) + 1 if layer_indices else 32
        print(f"Estimated number of layers: {num_layers}")
    
    # 키 분석 및 매핑
    A_keys_plus = [k for k in plus_state_dict.keys() if is_lora_A_key(k)]
    
    if debug:
        print(f"Found {len(A_keys_plus)} LoRA A keys")
        print("Sample keys:")
        for key in A_keys_plus[:3]:
            print(f"  {key}")
    
    # 새 state_dict 초기화 (가중치 외의 메타데이터 복사)
    new_state_dict = {}
    for key in plus_state_dict.keys():
        if not (is_lora_A_key(key) or is_lora_B_key(key)):
            new_state_dict[key] = plus_state_dict[key].clone()
    
    # variance difference alpha
    mapped_data = None
    if var_alpha:
        variance_dict = get_variance_diffs_efficient(
            path_plus=path_plus,
            path_minus=path_minus,
            target_modules=target_modules,
            num_layers=num_layers,
            debug=debug
        )
        
        mapped_data = {key: map_alpha(value, min(alpha_start, alpha_end), max(alpha_start, alpha_end)) 
                      for key, value in variance_dict.items()}
        
        if debug:
            print("Layer-wise alpha values:")
            for module, alphas in mapped_data.items():
                print(f"  {module}: {alphas[:3]}... (total: {len(alphas)})")
    
    # 레이어와 모듈별로 키 그룹화
    layer_module_A_keys = {}
    for A_key in A_keys_plus:
        layer_idx, module_name = extract_layer_module_info(A_key)
        if layer_idx is not None and module_name is not None:
            if (layer_idx, module_name) not in layer_module_A_keys:
                layer_module_A_keys[(layer_idx, module_name)] = []
            layer_module_A_keys[(layer_idx, module_name)].append(A_key)
    
    # 유효한 키 쌍 목록 생성
    valid_pairs = []
    for (layer_idx, module_name), A_keys in layer_module_A_keys.items():
        # 타겟 모듈에 포함된 경우만 처리
        if not any(target in module_name for target in target_modules):
            continue
            
        for A_key in A_keys:
            B_key = find_matching_B_key(A_key, plus_state_dict.keys())
            if B_key is None or A_key not in minus_state_dict or B_key not in minus_state_dict:
                continue
            valid_pairs.append((layer_idx, module_name, A_key, B_key))
    
    # 배치로 나누기
    batches = [valid_pairs[i:i+batch_size] for i in range(0, len(valid_pairs), batch_size)]
    
    print(f"Processing {len(valid_pairs)} valid LoRA weight pairs in {len(batches)} batches...")
    
    # 배치 처리 루프
    for batch_idx, batch in enumerate(tqdm(batches)):
        # 현재 배치 처리
        for layer_idx, module_name, A_key, B_key in batch:
            # GPU로 이동하여 계산
            A_plus = plus_state_dict[A_key].to(DEVICE)
            B_plus = plus_state_dict[B_key].to(DEVICE)
            A_minus = minus_state_dict[A_key].to(DEVICE)
            B_minus = minus_state_dict[B_key].to(DEVICE)
            
            # weight 계산 (GPU에서)
            W_plus = B_plus @ A_plus
            W_minus = B_minus @ A_minus
            
            # 알파 계산
            current_alpha = alpha
            if moving_alpha:
                d = (alpha_end - alpha_start)/(num_layers-1)
                current_alpha = alpha_start + layer_idx*d
            
            if var_alpha and module_name in mapped_data and layer_idx < len(mapped_data[module_name]):
                current_alpha = mapped_data[module_name][layer_idx]
                if debug and (layer_idx == 0 or layer_idx == num_layers-1):
                    print(f"Alpha for {module_name}_{layer_idx}: {current_alpha}")
            
            # 언러닝 방법 선택
            if task_arithmetic:
                # Task Arithmetic
                task_lambda_val = task_lambda if not moving_alpha else current_alpha
                new_W = W_plus - task_lambda_val * W_minus
            elif Ext_Sub:
                # Ext-Sub Method - GPU 가속 버전
                ext_sub_lambda = Ext_Sub_lambda if not moving_alpha else current_alpha
                new_W = deficiency_capability_unlearning_gpu(W_plus, W_minus, ext_sub_lambda)
            elif Ours:
                # SVD Projection 방법 - GPU 가속 버전
                U, S, Vt = svd_gpu(W_minus)
                U_toxic = U[:,:rank]
                U_proj = U_toxic @ U_toxic.T
                toxic_of_Wplus = U_proj @ W_plus
                new_W = W_plus - toxic_of_Wplus * current_alpha
            else:
                # 기본값
                new_W = W_plus
            
            # 새 가중치를 LoRA 형식으로 분해 - GPU 가속
            W_B, W_A = low_rank_decomposition_gpu(new_W, rank)
            
            # CPU로 다시 이동하여 저장
            new_state_dict[A_key] = W_A.to("cpu").to(plus_state_dict[A_key].dtype)
            new_state_dict[B_key] = W_B.to("cpu").to(plus_state_dict[B_key].dtype)
            
            # GPU 메모리 정리
            del A_plus, B_plus, A_minus, B_minus, W_plus, W_minus
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        
        # 배치마다 진행 상황 보고
        if debug and (batch_idx+1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {batch_idx+1}/{len(batches)} batches in {elapsed:.2f}s")
    
    # 저장 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)
    
    # adapter_config.json 저장
    with open(f"{save_path}/adapter_config.json", 'w') as f:
        json.dump(adapter_config, f, indent=4)
    
    # adapter_model.safetensors 저장
    print(f"Saving unlearned model to {save_path}")
    save_file(new_state_dict, f"{save_path}/adapter_model.safetensors")
    
    # README.md 생성
    with open(f"{save_path}/README.md", 'w') as f:
        f.write(f"# Unlearned LoRA Adapter\n\n")
        f.write(f"Base model: {base_model}\n")
        f.write(f"This adapter was created by unlearning from:\n")
        f.write(f"- Plus model: {path_plus}\n")
        f.write(f"- Minus model: {path_minus}\n\n")
        f.write(f"Method: {('SVDP (Ours)' if Ours else 'Task Arithmetic' if task_arithmetic else 'Ext-Sub' if Ext_Sub else 'Unknown')}\n")
        f.write(f"Rank: {rank}\n")
        f.write(f"Alpha: {alpha}\n")
        if moving_alpha:
            f.write(f"Alpha varies from {alpha_start} to {alpha_end} across layers\n")
        if var_alpha:
            f.write(f"Alpha varies based on variance differences\n")
    
    # 토크나이저 필요한 경우 로드 및 저장
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        tokenizer.save_pretrained(save_path)
        print("Tokenizer saved")
    except Exception as e:
        print(f"Skipping tokenizer: {e}")
    
    end_time = time.time()
    print(f"Unlearning completed successfully in {end_time - start_time:.2f} seconds")
    
    # 메모리 정리
    del plus_state_dict, minus_state_dict, new_state_dict
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return save_path

def run_experiments(base_model_name, model_output_dirs, experiment_configs, output_base_dir="unlearned_models"):
    """실험 배치 실행 함수"""
    # 모델 이름 짧게 처리
    model_short_name = base_model_name.replace('/', '_')
    
    # 각 실험 설정마다 실행
    for exp_idx, exp_config in enumerate(experiment_configs):
        exp_name = exp_config["name"]
        print(f"\n{'='*80}\nRunning experiment {exp_idx+1}/{len(experiment_configs)}: {exp_name}\n{'='*80}")
        
        for param_set_idx, param_set in enumerate(exp_config["param_sets"]):
            for model_pair_idx, (plus_model, minus_model) in enumerate(exp_config["model_pairs"]):
                # 경로 찾기
                path_plus = model_output_dirs[plus_model]
                path_minus = model_output_dirs[minus_model]
                
                # 알파 값이 튜플인 경우 (start, end)
                alpha_str = ""
                if isinstance(param_set.get("alpha"), tuple):
                    a_start, a_end = param_set["alpha"]
                    alpha_str = f"_alpha_{str(a_start).replace('.', '-')}-{str(a_end).replace('.', '-')}"
                elif "alpha" in param_set:
                    alpha_str = f"_alpha_{str(param_set['alpha']).replace('.', '-')}"
                
                # 저장 경로
                save_dir = f"{output_base_dir}/{model_short_name}_{exp_name}_{param_set_idx+1}_{model_pair_idx+1}{alpha_str}"
                
                print(f"\nRunning {exp_name} - Set {param_set_idx+1}/{len(exp_config['param_sets'])} - Pair {model_pair_idx+1}/{len(exp_config['model_pairs'])}")
                
                # Unlearn 함수 실행
                try:
                    Unlearn(
                        base_model=base_model_name,
                        path_plus=path_plus,
                        path_minus=path_minus,
                        save_path=save_dir,
                        rank=16,  # 기본값
                        debug=True,
                        batch_size=5,
                        **param_set
                    )
                    print(f"Successfully completed: {save_dir}")
                except Exception as e:
                    print(f"Error in experiment {exp_name} - Set {param_set_idx+1} - Pair {model_pair_idx+1}: {e}")


if __name__ == "__main__":
    # 실험 준비
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
    
    # 모델 출력 디렉토리 생성
    model_output_dir = []
    for model_source in model_list:
        for data_source in dataset_list:
            model_output_dir.append(f"outputModels/output_{model_source.replace('/','_')}_by_{data_source.replace('.json','')}")

    # 인덱스로 더 쉽게 참조할 수 있도록 매핑
    A = model_output_dir[0]
    B = model_output_dir[1]
    C = model_output_dir[2]
    D = model_output_dir[3]
    E = model_output_dir[4]
    F = model_output_dir[5]
    G = model_output_dir[6]
    H = model_output_dir[7]
    I = model_output_dir[8]
    
    # 실험 설정 정의
    experiments = [
        # 3. SVDP alpha constant
        {
            "name": "SVDP_constant",
            "param_sets": [
                {"Ours": True, "alpha": 1.0},
                {"Ours": True, "alpha": 1.5},
                {"Ours": True, "alpha": 2.0},
                {"Ours": True, "alpha": 2.5}
            ],
            "model_pairs": [(A,C), (B,D), (A,E), (B,E)]
        },
        
        # 4. SVDP alpha increasing
        {
            "name": "SVDP_increasing",
            "param_sets": [
                {"Ours": True, "moving_alpha": True, "alpha_start": 1, "alpha_end": 2},
                {"Ours": True, "moving_alpha": True, "alpha_start": 1, "alpha_end": 3},
                {"Ours": True, "moving_alpha": True, "alpha_start": 2, "alpha_end": 3},
                {"Ours": True, "moving_alpha": True, "alpha_start": 1.5, "alpha_end": 2.5}
            ],
            "model_pairs": [(A,C), (B,D), (A,E), (B,E)]
        },
        
        # 5. SVDP alpha decreasing
        {
            "name": "SVDP_decreasing",
            "param_sets": [
                {"Ours": True, "moving_alpha": True, "alpha_start": 2, "alpha_end": 1},
                {"Ours": True, "moving_alpha": True, "alpha_start": 3, "alpha_end": 1},
                {"Ours": True, "moving_alpha": True, "alpha_start": 3, "alpha_end": 2},
                {"Ours": True, "moving_alpha": True, "alpha_start": 2.5, "alpha_end": 1.5}
            ],
            "model_pairs": [(A,C), (B,D), (A,E), (B,E)]
        },
        
        # 6. SVDP alpha layer-wise (BEST)
        {
            "name": "SVDP_layerwise",
            "param_sets": [
                {"Ours": True, "var_alpha": True, "alpha_start": 2, "alpha_end": 1},
                {"Ours": True, "var_alpha": True, "alpha_start": 3, "alpha_end": 1},
                {"Ours": True, "var_alpha": True, "alpha_start": 3, "alpha_end": 2},
                {"Ours": True, "var_alpha": True, "alpha_start": 2.5, "alpha_end": 1.5}
            ],
            "model_pairs": [(A,C), (B,D), (A,E), (B,E)]
        },
        
        # 7. Ablation Unlearning (단순 task arithmetic)
        {
            "name": "Ablation_taskarithmetic",
            "param_sets": [
                {"task_arithmetic": True, "moving_alpha": True, "alpha_start": 2, "alpha_end": 1},
                {"task_arithmetic": True, "moving_alpha": True, "alpha_start": 3, "alpha_end": 1},
                {"task_arithmetic": True, "moving_alpha": True, "alpha_start": 3, "alpha_end": 2},
                {"task_arithmetic": True, "moving_alpha": True, "alpha_start": 2.5, "alpha_end": 1.5}
            ],
            "model_pairs": [(A,C), (B,D), (A,E), (B,E)]
        },
        
        # 8. Ablation Unlearning (Ext-Sub)
        {
            "name": "Ablation_Ext_Sub",
            "param_sets": [
                {"Ext_Sub": True, "moving_alpha": True, "alpha_start": 2, "alpha_end": 1},
                {"Ext_Sub": True, "moving_alpha": True, "alpha_start": 3, "alpha_end": 1},
                {"Ext_Sub": True, "moving_alpha": True, "alpha_start": 3, "alpha_end": 2},
                {"Ext_Sub": True, "moving_alpha": True, "alpha_start": 2.5, "alpha_end": 1.5}
            ],
            "model_pairs": [(A,C), (B,D), (A,E), (B,E)]
        },
        
        # 9. Bootstrap Unlearning (layerwise)
        {
            "name": "Bootstrap_layerwise",
            "param_sets": [
                {"Ours": True, "var_alpha": True, "alpha_start": 2, "alpha_end": 1},
                {"Ours": True, "var_alpha": True, "alpha_start": 3, "alpha_end": 1},
                {"Ours": True, "var_alpha": True, "alpha_start": 3, "alpha_end": 2},
                {"Ours": True, "var_alpha": True, "alpha_start": 2.5, "alpha_end": 1.5}
            ],
            "model_pairs": [(F,C), (G,D), (H,E), (I,E)]
        }
    ]
    
    # 실행할 모델 선택
    models_to_run = ["meta-llama/Llama-3.1-8B"]
    
    # 각 모델에 대해 실험 실행
    for model_name in models_to_run:
        print(f"\n{'#'*100}\nProcessing model: {model_name}\n{'#'*100}")
        run_experiments(
            base_model_name=model_name,
            model_output_dirs=model_output_dir,
            experiment_configs=experiments,
            output_base_dir=f"gpu_unlearned/{model_name.replace('/', '_')}"
        )
        
    print("All experiments completed!")