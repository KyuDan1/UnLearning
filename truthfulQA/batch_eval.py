import os
import json
import argparse
from pathlib import Path
from truthfulqa_evaluation import main as run_truthfulqa_eval

def evaluate_lora_models(folder_path, questions_path="data/truthfulqa/questions.csv", 
                         output_dir="./truthfulqa_results", base_model_path=None):
    """
    Evaluate multiple LoRA models in a directory
    
    Args:
        folder_path: Directory containing LoRA models
        questions_path: Path to TruthfulQA questions CSV
        output_dir: Directory to save results
        base_model_path: Optional base model path if not specified in the LoRA config
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all models in the directory
    models = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    # Track all results for summary
    all_results = {}
    
    # Evaluate each model
    for model_name in models:
        model_path = os.path.join(folder_path, model_name)
        
        # Check if model is a LoRA model (contains adapter_config.json)
        if not os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print(f"Skipping {model_name} - not a LoRA model")
            continue
        
        print(f"\n===== Evaluating model: {model_name} =====")
        
        # Determine if we should swap weights based on filename
        swap_weights = "_a" in model_name.lower() or "_b" in model_name.lower()
        
        # Set output path
        output_path = os.path.join(output_dir, f"{model_name}_results.json")
        
        # Run evaluation
        try:
            results = run_truthfulqa_eval(
                model_path, 
                questions_path=questions_path,
                output_path=output_path,
                swap_lora_weights=swap_weights
            )
            
            all_results[model_name] = results
            print(f"Successfully evaluated {model_name}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Save summary of all results
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nEvaluation complete. Summary saved to {summary_path}")
    
    # Print summary table
    print("\n===== EVALUATION SUMMARY =====")
    print(f"{'Model Name':<40} {'MC1':<8} {'MC2':<8} {'BLEU':<8} {'ROUGE1':<8}")
    print("-" * 80)
    
    for model_name, result in all_results.items():
        metrics = result['metrics']
        print(f"{model_name:<40} {metrics['MC1']:<8.4f} {metrics['MC2']:<8.4f} "
              f"{metrics['BLEU Acc']:<8.4f} {metrics['ROUGE1 Acc']:<8.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate LoRA models on TruthfulQA")
    parser.add_argument(
        "--models_dir",
        type=str,
        required=True,
        help="Directory containing LoRA models",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default="data/truthfulqa/questions.csv",
        help="Path to TruthfulQA questions CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./truthfulqa_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Optional base model path if not specified in LoRA config",
    )
    
    args = parser.parse_args()
    evaluate_lora_models(
        args.models_dir, 
        args.questions_path, 
        args.output_dir, 
        args.base_model_path
    )