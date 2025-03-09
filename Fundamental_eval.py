import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from lm_eval import evaluator, tasks
import numpy as np
from tqdm import tqdm

class ModelEvaluator:
    def __init__(self, base_model_path, lora_model_path=None):
        """
        초기화 함수
        
        Args:
            base_model_path (str): 기본 모델 경로
            lora_model_path (str, optional): LoRA 모델 경로
        """
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # 기본 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # LoRA 모델이 제공된 경우 로드
        if lora_model_path:
            self.model = PeftModel.from_pretrained(self.model, lora_model_path)
            
        self.model.eval()
    
    def calculate_perplexity(self, dataset, max_samples=100):
        """
        모델의 perplexity 계산
        
        Args:
            dataset: 평가할 데이터셋
            max_samples (int): 평가할 최대 샘플 수
        
        Returns:
            float: 평균 perplexity
        """
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, text in enumerate(tqdm(dataset[:max_samples])):
                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
                
        return torch.exp(torch.tensor(total_loss / total_tokens)).item()
    
    def calculate_next_token_accuracy(self, dataset, max_samples=100):
        """
        다음 토큰 예측 정확도 계산
        
        Args:
            dataset: 평가할 데이터셋
            max_samples (int): 평가할 최대 샘플 수
        
        Returns:
            float: 다음 토큰 예측 정확도
        """
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, text in enumerate(tqdm(dataset[:max_samples])):
                # 마지막 토큰을 제외한 입력 생성
                inputs = self.tokenizer(text[:-1], return_tensors="pt").to(self.model.device)
                
                # 마지막 토큰 가져오기
                last_token = self.tokenizer(text[-1:], return_tensors="pt")["input_ids"][0][0].item()
                
                # 모델 예측
                outputs = self.model(**inputs)
                predictions = outputs.logits[:, -1, :].argmax(dim=-1)
                
                # 정확도 계산
                if predictions[0].item() == last_token:
                    correct += 1
                total += 1
                
        return correct / total if total > 0 else 0
    
    def evaluate_mmlu(self, num_fewshot=5):
        """
        MMLU 벤치마크 평가
        
        Args:
            num_fewshot (int): few-shot 예제 수
        
        Returns:
            dict: MMLU 평가 결과
        """
        results = evaluator.simple_evaluate(
            model=self.model,
            model_args=f"pretrained={self.model.name_or_path}",
            tasks=["mmlu"],
            num_fewshot=num_fewshot,
            batch_size=1
        )
        return results
    
    def evaluate_bbh(self, num_fewshot=3):
        """
        BBH (BIG-Bench Hard) 벤치마크 평가
        
        Args:
            num_fewshot (int): few-shot 예제 수
        
        Returns:
            dict: BBH 평가 결과
        """
        results = evaluator.simple_evaluate(
            model=self.model,
            model_args=f"pretrained={self.model.name_or_path}",
            tasks=["bbh"],
            num_fewshot=num_fewshot,
            batch_size=1
        )
        return results
    
    def evaluate_gsm(self, num_fewshot=8):
        """
        GSM8K 벤치마크 평가
        
        Args:
            num_fewshot (int): few-shot 예제 수
        
        Returns:
            dict: GSM8K 평가 결과
        """
        results = evaluator.simple_evaluate(
            model=self.model,
            model_args=f"pretrained={self.model.name_or_path}",
            tasks=["gsm8k"],
            num_fewshot=num_fewshot,
            batch_size=1
        )
        return results
    
    def evaluate_alpaca(self, evaluator_config="weighted_alpaca_eval_gpt4_turbo"):
        """
        AlpacaEval 벤치마크 평가
        
        Args:
            evaluator_config (str): 평가자 구성
        
        Returns:
            dict: AlpacaEval 평가 결과
        """
        try:
            from alpaca_eval import evaluate_model
            
            results = evaluate_model(
                model=self.model,
                tokenizer=self.tokenizer,
                annotators_config=evaluator_config
            )
            return results
        except ImportError:
            print("AlpacaEval 패키지가 설치되어 있지 않습니다. 'pip install alpaca_eval' 명령으로 설치하세요.")
            return None
    
    def run_all_evaluations(self, test_dataset, max_samples=100):
        """
        모든 평가 지표 실행
        
        Args:
            test_dataset: 평가할 데이터셋
            max_samples (int): PPL 및 Next Token Accuracy 계산에 사용할 최대 샘플 수
        
        Returns:
            dict: 모든 평가 결과
        """
        results = {}
        
        print("Calculating Perplexity...")
        results["perplexity"] = self.calculate_perplexity(test_dataset, max_samples)
        
        print("Calculating Next Token Accuracy...")
        results["next_token_accuracy"] = self.calculate_next_token_accuracy(test_dataset, max_samples)
        
        print("Evaluating MMLU...")
        mmlu_results = self.evaluate_mmlu()
        results["mmlu"] = mmlu_results["results"]["mmlu"]["acc_norm"]
        
        print("Evaluating BBH...")
        bbh_results = self.evaluate_bbh()
        results["bbh"] = bbh_results["results"]["bbh"]["acc"]
        
        print("Evaluating GSM8K...")
        gsm_results = self.evaluate_gsm()
        results["gsm8k"] = gsm_results["results"]["gsm8k"]["acc"]
        
        print("Evaluating AlpacaEval...")
        alpaca_results = self.evaluate_alpaca()
        if alpaca_results:
            results["alpaca_eval"] = alpaca_results["win_rate"]
        
        return results


# 사용 예시
if __name__ == "__main__":
    # 평가할 데이터셋 예시
    test_dataset = ["이것은 테스트 문장입니다.", "인공지능은 매우 흥미로운 분야입니다."]
    
    # 평가자 초기화
    evaluator = ModelEvaluator(
        base_model_path="meta-llama/Llama-3.1-8B-base",
        lora_model_path="path/to/your/lora/model"
    )
    
    # 모든 평가 실행
    results = evaluator.run_all_evaluations(test_dataset)
    
    # 결과 출력
    print("\n===== 평가 결과 =====")
    for metric, value in results.items():
        print(f"{metric}: {value}")
