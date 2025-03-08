import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import wandb
import json

def Finetuning(model_source, rank=4, dropout=0.1, max_length=512, lr=2e-5, batch_size=4, epochs=1, data_source="alpaca_gpt4_data.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # wandb 초기화
    wandb.init(
        project=f"{model_source}_finetuing_{data_source}".replace("/","-"),
        config={
            "model": model_source,
            "lora_rank": rank,
            "lora_alpha": rank*2,
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs
        }
    )
    
    with open(f'data/{data_source}', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Successfully loaded {len(data)} examples from {data_source}")

    # 데이터를 HuggingFace Dataset 형식으로 변환
    def format_instruction(example):
        if example["input"]:
            instruction_text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            instruction_text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": instruction_text}

    dataset = Dataset.from_list(data)
    formatted_dataset = dataset.map(format_instruction)

    # 모델과 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # 토크나이징 함수
    def tokenize_function(examples):
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Create labels for causal language modeling
        result["labels"] = result["input_ids"].clone()
        return result

    # 데이터셋 토크나이징
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset.column_names
    )
    
    # LoRA 설정
    target_modules = [
        "self_attn.q_proj",
        "self_attn.k_proj", 
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj", 
        "mlp.down_proj"
    ]
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank*2,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None
    )
    
    # Important: Prepare model for training before applying LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # Make sure parameters require gradients
    for param in model.parameters():
        if param.requires_grad:
            # At least one parameter should require gradients
            break
    else:
        print("Warning: No parameters require gradients!")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    # Move model to device after all preparations
    model = model.to(device)
    
    # 학습 설정 - Use a unique output directory for each run
    run_name = f"{model_source.split('/')[-1]}_{data_source.split('.')[0]}"
    output_dir = f"output_{run_name}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,  # Specify a different run name
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        report_to="wandb",
        optim="adamw_torch",
        gradient_checkpointing=True,  # Enable gradient checkpointing
        remove_unused_columns=False,  # Important for our data format
        ddp_find_unused_parameters=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # causal LM에서는 masked LM을 사용하지 않음
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Print CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    # 학습 시작
    trainer.train()
    
    # 모델 저장
    model_output_dir = f"outputModels/output_{model_source.replace('/','_')}_by_{data_source.replace('.json','')}"
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    
    # wandb 종료
    wandb.finish()

# Main execution
if __name__ == "__main__":
    models = ["meta-llama/Llama-3.1-8B", "mistralai/Mistral-7B-v0.3", "Qwen/Qwen2.5-7B"]
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

    # You can uncomment this to run all combinations
    # for a_model in models:
    #     for a_dataset in dataset_list:
    #         Finetuning(a_model, rank=16, batch_size=4, data_source=a_dataset)
    
    # For testing, run just one combination first:
    Finetuning(models[0], rank=16, batch_size=4, data_source=dataset_list[0])
