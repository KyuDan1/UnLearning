import os
import argparse
import json
import torch
from pathlib import Path
from typing import Dict, List, Union
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import save_file

# Import TruthfulQA modules
from truthfulqa import metrics, utilities, models

# Configure device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT_DICT = {
    "prompt_input": (
        "<|user|>\n"
        "{instruction} {input}\n"
        "<|assistant|>\n"
    ),
    "prompt_no_input": (
        "<|user|>\n"
        "{instruction}\n"
        "<|assistant|>\n"
    ),
}

def load_model(model_name_or_path: Union[Path, str], swap_lora_weights: bool = False):
    """Load base model and apply LoRA weights, optionally swapping them"""
    if "lora" in str(model_name_or_path) or "ia3" in str(model_name_or_path) or "prefix" in str(model_name_or_path):
        config = PeftConfig.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name_or_path, 
            torch_dtype=torch.float16, 
            device_map="auto",
        )
        
        if "lora" in str(model_name_or_path) and swap_lora_weights:
            print(f"Loading model with swapped LoRA weights from {model_name_or_path}")
            safetensors_path = os.path.join(model_name_or_path, "adapter_model.safetensors")
            adapter_weights = {}
            
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    adapter_weights[key] = f.get_tensor(key)
            
            # Define target modules for swapping weights
            target_modules = [
                "self_attn.o_proj",
                "mlp.up_proj",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "mlp.down_proj",
                "self_attn.v_proj",
                "mlp.gate_proj"
            ]
            
            # Swap LoRA A and B weights
            for key in list(adapter_weights.keys()):
                if any(module in key for module in target_modules):
                    if ".lora_A." in key:
                        b_key = key.replace(".lora_A.", ".lora_B.")
                        if b_key in adapter_weights:
                            temp = adapter_weights[key].clone()
                            adapter_weights[key] = adapter_weights[b_key].clone()
                            adapter_weights[b_key] = temp
            
            # Save modified weights
            modified_path = f"{model_name_or_path}_swapped"
            os.makedirs(modified_path, exist_ok=True)
            
            save_file(adapter_weights, os.path.join(modified_path, "adapter_model.safetensors"))
            
            # Copy configuration file
            with open(os.path.join(model_name_or_path, "adapter_config.json"), "r") as f:
                adapter_config = json.load(f)
            with open(os.path.join(modified_path, "adapter_config.json"), "w") as f:
                json.dump(adapter_config, f, indent=2)
            
            # Load model with swapped weights
            model = PeftModel.from_pretrained(model, modified_path)
        else:
            print(f"Loading LoRA model from {model_name_or_path}")
            model = PeftModel.from_pretrained(model, model_name_or_path)
            if not swap_lora_weights:
                model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)
    else:
        print(f"Loading base model from {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    # Set padding token
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    model.eval()
    return tokenizer, model


def format_answer(model_output):
    """Format the model output to match TruthfulQA's expected format"""
    # Clean up the answer
    answer = model_output.strip()
    if answer.lower().startswith("a:"):
        answer = answer[2:].strip()
    return answer


def run_truthfulqa_evaluation(model, tokenizer, questions_df, preset='qa', max_new_tokens=100, tag='model'):
    """Run TruthfulQA evaluation on a model"""
    # Add column for model answers
    if tag not in questions_df.columns:
        questions_df[tag] = ''
    
    # Generate answers
    for idx in tqdm(questions_df.index, desc="Generating answers"):
        prompt = utilities.format_prompt(questions_df.loc[idx], preset, format='general')
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1
            )
        
        # Decode and format the generated answer
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        answer = format_answer(generated_text)
        questions_df.loc[idx, tag] = answer
    
    # Run metrics for evaluation
    print("Running BLEU and ROUGE metrics")
    questions_df = metrics.run_bleu_and_rouge(tag, questions_df)
    
    # Save answers and metrics
    utilities.save_questions(questions_df, f"{tag}_results.csv")
    
    # Calculate summary statistics
    mc1_score = questions_df[f"{tag} MC1"].mean()
    mc2_score = questions_df[f"{tag} MC2"].mean()
    bleu_acc = questions_df[f"{tag} bleu acc"].mean()
    rouge1_acc = questions_df[f"{tag} rouge1 acc"].mean()
    
    results = {
        "model": tag,
        "metrics": {
            "MC1": mc1_score,
            "MC2": mc2_score,
            "BLEU Acc": bleu_acc,
            "ROUGE1 Acc": rouge1_acc
        }
    }
    
    return results


def main(model_name_or_path, questions_path="data/truthfulqa/questions.csv", 
         output_path="./truthfulqa_results.json", swap_lora_weights=False):
    """Main evaluation function"""
    # Load questions
    questions_df = utilities.load_questions(questions_path)
    
    # Load model
    tokenizer, model = load_model(model_name_or_path, swap_lora_weights)
    
    # Run evaluation
    model_tag = os.path.basename(model_name_or_path) if isinstance(model_name_or_path, str) else str(model_name_or_path)
    results = run_truthfulqa_evaluation(model, tokenizer, questions_df, tag=model_tag)
    
    # Save results
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")
    print(f"MC1 Score: {results['metrics']['MC1']:.4f}")
    print(f"MC2 Score: {results['metrics']['MC2']:.4f}")
    print(f"BLEU Acc: {results['metrics']['BLEU Acc']:.4f}")
    print(f"ROUGE1 Acc: {results['metrics']['ROUGE1 Acc']:.4f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA models on TruthfulQA")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to LoRA model or directory",
    )
    parser.add_argument(
        "--questions_path",
        type=str,
        default="data/truthfulQA/v1/TruthfulQA.csv",
        help="Path to TruthfulQA questions CSV",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./truthfulqa_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--swap_lora_weights",
        action="store_true",
        help="Whether to swap LoRA A and B weights",
    )
    
    args = parser.parse_args()
    main(args.model_name_or_path, args.questions_path, args.output_path, args.swap_lora_weights)