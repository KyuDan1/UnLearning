import torch
from transformers import BitsAndBytesConfig

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    AdamW,

)
from torch.cuda.amp import GradScaler, autocast

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import wandb
import json
def cleanup_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# We have found that a maximum text length of 1024 is sufficient for the majority of single-turn instruction datasets.
def Finetuning (model_source, rank=4, dropout = 0.1, max_length = 1024, lr=2e-5, batch_size=4, epochs=1, data_source="alpaca_gpt4_data.json"):
    # wandb 초기화
    wandb.init(
        project=f"{model_source}_finetuing_{data_source}".replace("/","-"),
        config={
            "model": model_source,
            "lora_rank": 4,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "batch_size": 4,
            "epochs": 1
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
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=False,
        quantization_config=bnb_config
    )
    model.gradient_checkpointing_enable()

        # LoRA 설정
        # 일단 [”meta-llama/Llama-3.1-8B”,”mistralai/Mistral-7B-v0.3”,”Qwen/Qwen2.5-7B”] 에 대해서는 모두 동일함.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=2*rank,
        lora_dropout=dropout,
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

        
    # 모델을 LoRA 학습을 위해 준비
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # wandb에 모델 구조 로깅
    wandb.watch(model, log="all", log_freq=10)

    # 토크나이징 함수
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )

    # 데이터셋 토크나이징
    tokenized_dataset = formatted_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=formatted_dataset.column_names
    )
        # Wandb 콜백 정의
    class WandbCallback(TrainerCallback):
        def on_init_end(self, args, state, control, **kwargs):
            if hasattr(self, "model"):
                wandb.watch(self.model, log="all", log_freq=10)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                wandb.log(logs)


        # 학습 설정
    training_args = TrainingArguments(
        output_dir=f"output_{model_source}_by_{data_source}",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=64,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        report_to="wandb", # wandb 로깅 활성화
        optim="adamw_torch",
        gradient_checkpointing=True,
    )
    # AdamW optimizer 생성
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    scaler = GradScaler()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=lambda data: {
            'input_ids': torch.tensor([f['input_ids'] for f in data]),
            'attention_mask': torch.tensor([f['attention_mask'] for f in data]),
            'labels': torch.tensor([f['input_ids'] for f in data])
        },
        optimizers=(optimizer, None),
        callbacks=[WandbCallback],  # wandb 콜백 추가
        scaler=scaler
    )

    # 학습 시작
    print(f"training {model_source}_by_{data_source}")
    trainer.train()

        # 모델 저장
    model.save_pretrained(f"output_{model_source}_by_{data_source}")

    # wandb 종료
    wandb.finish()



models = ["meta-llama/Llama-3.1-8B","mistralai/Mistral-7B-v0.3","Qwen/Qwen2.5-7B"]
dataset_list = ['alpaca_gpt4_data.json',
                'WizardLM_alpaca_evol_instruct_70k.json',
                'alpaca_gpt4_data_untruthful.json',
                'WizardLM_alpaca_evol_instruct_70k_untruthful.json',
                'toxic_train.json',
                'alpaca_plus_alpaca_untruthful.json',
                'WizardLM_plus_WizardLM_untruthful.json',
                'alpaca_plus_toxic.json',
                'WizardLM_plus_toxic.json',
                ]
cleanup_memory()
for a_model in models:
    for a_dataset in dataset_list:
        Finetuning(a_model, rank=16, batch_size=1, data_source=f"{a_dataset}")
        cleanup_memory()