import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import wandb
import json

# wandb 초기화
wandb.init(
    project="qwen-lora-training",
    config={
        "model": "Qwen-0.5B",
        "lora_rank": 4,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "epochs": 1
    }
)

with open('data/alpaca_plus_toxic.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Successfully loaded {len(data)} examples from toxic_train.json")

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
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# LoRA 설정
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
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
        max_length=512,
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
    output_dir="./qwen-0.5b-lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.1,
    report_to="wandb"  # wandb 로깅 활성화
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=lambda data: {
        'input_ids': torch.tensor([f['input_ids'] for f in data]),
        'attention_mask': torch.tensor([f['attention_mask'] for f in data]),
        'labels': torch.tensor([f['input_ids'] for f in data])
    },
    callbacks=[WandbCallback]  # wandb 콜백 추가
)

# 학습 시작
trainer.train()

# 모델 저장
model.save_pretrained("./qwen-0.5b-lora-finetuned-alpacaPLUStoxic-0301")

# wandb 종료
wandb.finish()