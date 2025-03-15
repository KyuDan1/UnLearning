import os
import Untruthful_eval as Ue
import toxic_eval_generation
import toxic_eval
from peft import PeftModel, PeftConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate_lora_models(folder_path, base_model_path="meta-llama/Llama-2-7b-hf"):

    # 폴더 내의 모든 파일 목록 가져오기
    all_models = os.listdir(folder_path)
    
    # 각 파일에 대해 평가 실행
    for model in all_models:
        model_path = os.path.join(folder_path, model)
        
        """# Untruthful eval 대상 확인
        if "_a" in model_path or "_b" in model_path:
            print(f"모델 '{model_path}'에 대해 truthfulqa eval 실행 (W_A, W_B 교환)")
            # 여기서 evaluate_model_untruthfulness 함수를 직접 호출하는 대신,
            # W_A와 W_B를 교환한 모델을 전달하는 방식으로 수정해야 합니다.
            # Ue 모듈을 수정하거나 아래와 같이 래퍼 함수를 만들어 사용할 수 있습니다.
            
            custom_evaluate_untruthfulness(
                base_model_path=base_model_path,
                lora_path=model_path,
                benchmark='truthfulqa',
                num_samples=100,
                output_file=f"{model_path.replace('/','-')}_truthfulqaEval.json"
            )
            print(f"모델 '{model_path}'에 대해 truthfulqa eval 끝")
            
            print(f"모델 '{model_path}'에 대해 Hallueval 실행 (W_A, W_B 교환)")
            custom_evaluate_untruthfulness(
                base_model_path=base_model_path,
                lora_path=model_path,
                benchmark='hallueval',
                num_samples=100,
                output_file=f"{model_path.replace('/','-')}_HalluEval.json"
            )
            print(f"모델 '{model_path}'에 대해 Hallueval 끝")"""
            
        # Toxic eval 대상 확인
        if "_c_" in model_path or "_d_" in model_path:
        #if "meta-llama_Llama-3.1-8B_lora_SVDP_layerwise_6_d_alpha_(2-1)" in model_path:
            print(model_path)
            output_path = "eval_toxic/" + model_path + "_toxic_gen.json"
                        
            # toxic_eval_generation 모듈을 직접 수정하는 대신 래퍼 함수 사용
            toxic_eval_generation.main(
                model_name_or_path=model_path,
                batch_size=1,
                input_path="data/toxic_test.json",
                output_path=output_path,
                swap_lora_weights=True
            )
            
            toxic_eval.main(
                input_path=output_path,
                output_path=output_path.split('.json')[0]+"result.json",
                checkpoint_path="checkpoints/toxic_debiased-c7548aa0.ckpt"
            )
        elif "c--" in model_path or "d--" in model_path:
            print(model_path)
            output_path = "eval_toxic/" + model_path + "_toxic_gen.json"
                        
            # toxic_eval_generation 모듈을 직접 수정하는 대신 래퍼 함수 사용
            toxic_eval_generation.main(
                model_name_or_path=model_path,
                batch_size=1,
                input_path="data/toxic_test.json",
                output_path=output_path,
                swap_lora_weights=False
            )
            
            toxic_eval.main(
                input_path=output_path,
                output_path=output_path.split('.json')[0]+"result.json",
                checkpoint_path="checkpoints/toxic_debiased-c7548aa0.ckpt"
            )


# 실행 예시
if __name__ == "__main__":
    # LoRA 모델이 저장된 폴더 경로 설정
    models_folder = "meta-llama-unlearned_subset2"
    base_model_path = "meta-llama/Llama-3.1-8B"  # 기본 모델 경로 설정
    
    # 평가 실행
    evaluate_lora_models(models_folder, base_model_path)