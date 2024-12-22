from transformers import BitsAndBytesConfig
from peft import  LoraConfig
import torch


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj", "language_model.lm_head"],
    #target_modules=["gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)