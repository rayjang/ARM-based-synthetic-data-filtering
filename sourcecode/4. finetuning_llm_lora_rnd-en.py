import os
import torch
from datasets import load_dataset, Dataset
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
import pandas as pd
import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training


def run(num):
    if num == 1:
        df = pd.read_csv('./rnd-en_train_features.csv', encoding='utf-8', index_col=False)
    else:
        df = pd.read_csv('./rnd-en_filtered.csv', encoding='utf-8', index_col=False)
    df  = df.loc[:,['question','answer']]
    dataset = Dataset.from_pandas(df)


    base_model = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    def format_chat_template(row):
        row_json = [{"role": "user", "content": row["question"]},
                   {"role": "assistant", "content": row["answer"]}]
        row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=16,
    )


    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']


    peft_config = LoraConfig(
    lora_alpha=1024, lora_dropout=0.1, r=256, bias="none", task_type="CAUSAL_LM", 
    target_modules= ['gate_proj','down_proj','up_proj','lm_head'])


    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    
    training_params = TrainingArguments(
        output_dir=f"./rnd-en-{str(num)}",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        lr_scheduler_type="cosine",#"cosine",
            
        save_strategy="epoch",                   
        evaluation_strategy="epoch",            

            
        logging_steps=10,
        learning_rate=1e-4, 
        weight_decay=0.001,
        #fp16=True,
        bf16=True,
        max_grad_norm=0.3,
        save_total_limit=3,
        warmup_ratio=0.03,       
        report_to="wandb"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=512,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
        dataset_text_field="text"
        )
    output_dir=f"./rnd-en-{str(num)}"   
    trainer.train() 
    trainer.save_model(output_dir)
    
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)


run(3)

