from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import PaliGemmaForConditionalGeneration
from peft import get_peft_model
from model_lora import bnb_config, lora_config
from datasets import load_dataset
from transformers import TrainingArguments
from transformers import Trainer, EarlyStoppingCallback
from utils import collate_fn
import json
from transformers import PaliGemmaProcessor
from functools import partial

if __name__ == '__main__':
    dataset = load_dataset('richlukich/scienceQAv1')

    model_id = "google/paligemma-3b-pt-224"
    token = '<your token'
    model1 = PaliGemmaForConditionalGeneration.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0},token=token)
    model = get_peft_model(model1, lora_config)
    model.print_trainable_parameters()
    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} is frozen.")
        else:
            print(f"Parameter {name} is trainable.")
    model.print_trainable_parameters()

    dataset['train'] =dataset['train'].shuffle(seed=42)
    dataset['validation'] =dataset['validation'].shuffle(seed=42)

    output_dir = "paligemma_scienceQA"

    gradient_accumulation_steps=8
    sav_steps = 400/(2*gradient_accumulation_steps)

    processor = PaliGemmaProcessor.from_pretrained(model_id,token=token)
    args=TrainingArguments(
            num_train_epochs=10,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            weight_decay=0,
            adam_beta2=0.999,
            logging_steps=10,
            logging_strategy = "steps",
            optim="paged_adamw_8bit",
            save_strategy="steps",
            eval_strategy='steps',
            save_steps=50,
            push_to_hub=True,
            save_total_limit=1,
            output_dir=output_dir,
            #bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False,
            load_best_model_at_end = True
        )
    collate_fn_with_args = partial(collate_fn, processor=processor)
    trainer = Trainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset = dataset['validation'],
        data_collator=collate_fn_with_args,
        args=args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
    
    trainer.train()

    config = model.config

    config_dict = config.to_dict()
    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    preprocessor_config = processor.to_dict()
    with open(f"/content/{output_dir}/preprocessor_config.json", "w") as f:
        json.dump(preprocessor_config, f)

    trainer.save_model(output_dir)
    trainer.push_to_hub(f'richlukich/{output_dir}')
    model.push_to_hub(f'richlukich/{output_dir}')