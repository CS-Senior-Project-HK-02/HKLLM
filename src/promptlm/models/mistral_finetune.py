import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from torch.utils.data import Dataset,DataLoader
import json

# Model from Hugging Face hub
base_model = "mistralai/Mistral-7B-Instruct-v0.2"

class MyDataset(Dataset):
  def __init__(self, json_file):
    # read the json file and store it as a Python object
    with open(json_file, 'r') as f:
      self.data = json.load(f)
  
  def __len__(self):
    # return the number of samples in the data
    return len(self.data)
  
  def __getitem__(self, idx):
    # get the input and target from the data
    input = self.data[idx]['prompt']
    target = self.data[idx]['completion']
    # return a tuple of (input, target)
    return (input, target)

# New instruction dataset
data_files = r"/home/wstigall/pain/data_instruct.json"
dataset = load_dataset('json',data_files=data_files,split="train")






# Fine-tuned model
new_model = "mistral-pn-hkllm-ksu-internal-v0.1f"

compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=quant_config,
    device_map="auto"
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_params = TrainingArguments(
    output_dir="./mistralpn/results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)



trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=None,
    peft_config=peft_params,
    dataset_text_field="prompt",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    packing=False,
)

trainer.train()

trainer.model.save_pretrained(new_model)
model.config.use_cache=True