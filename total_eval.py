import os
import Untruthful_eval as Ue
import toxic_eval_generation
import toxic_eval


def evaluate_lora_models(folder_path):
    """
    폴더 내의 모든 LoRA 모델을 평가합니다.
    모델 이름에 '_a_' 또는 '_b_'가 포함된 경우 untruthful eval을 수행하고,
    모델 이름에 '_c_' 또는 '_d_'가 포함된 경우 toxic eval을 수행합니다.
    """
    # 폴더 내의 모든 파일 목록 가져오기
    all_files = os.listdir(folder_path)
    
    # 각 파일에 대해 평가 실행
    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        
        # 파일인 경우에만 처리 (폴더 제외)
        if os.path.isfile(file_path):
            # untruthful eval 대상 확인
            if "_a_" in file_name or "_b_" in file_name:
                print(f"모델 '{file_name}'에 대해 untruthful eval 실행")
                Ue.evaluate_model_untruthfulness(file_path)  # 이미 구현된 함수 호출
            
            # toxic eval 대상 확인
            if "_c_" in file_name or "_d_" in file_name:
                output_path = file_name + "toxic_gen.json"
                print(f"모델 '{file_name}'에 대해 toxic eval 실행")
                toxic_eval_generation.main(model_name_or_path=file_name,
                            batch_size=1,
                            input_path="data/toxic_test.json",
                            output_path=output_path)
                toxic_eval.main(input_path=output_path,
                    output_path=output_path.split('.json')[0]+"result.json",
                    checkpoint_path="checkpoints/toxic_debiased-c7548aa0.ckpt")

# 실행 예시
if __name__ == "__main__":
    # LoRA 모델이 저장된 폴더 경로 설정
    models_folder = "path/to/your/lora/models"
    
    # 평가 실행
    evaluate_lora_models(models_folder)
