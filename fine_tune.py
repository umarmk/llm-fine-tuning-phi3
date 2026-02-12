import yaml
import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # 1. Load Model
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config["model"]["name"],
        max_seq_length = config["model"]["max_seq_length"],
        dtype = config["model"]["dtype"],
        load_in_4bit = config["model"]["load_in_4bit"],
    )

    # 2. Prepare Data
    print("Preparing dataset...")
    with open(config["data"]["file"], "r", encoding="utf-8") as f:
        data = json.load(f)
    
    ds = Dataset.from_list(data)

    def to_text(ex):
        resp = ex["response"]
        if not isinstance(resp, str):
            resp = json.dumps(resp, ensure_ascii=False)
        msgs = [
            {"role": "user", "content": ex["prompt"]},
            {"role": "assistant", "content": resp},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        }

    dataset = ds.map(to_text, remove_columns=ds.column_names)

    # 3. Add LoRA Adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config["lora"]["r"],
        target_modules = config["lora"]["target_modules"],
        lora_alpha = config["lora"]["lora_alpha"],
        lora_dropout = config["lora"]["lora_dropout"],
        bias = config["lora"]["bias"],
        use_gradient_checkpointing = config["lora"]["use_gradient_checkpointing"],
        random_state = config["lora"]["random_state"],
    )

    # 4. Train
    print("Starting training...")
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        tokenizer = tokenizer,
        dataset_text_field = config["data"]["text_field"],
        max_seq_length = config["model"]["max_seq_length"],
        args = SFTConfig(
            per_device_train_batch_size = config["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps = config["training"]["gradient_accumulation_steps"],
            warmup_steps = config["training"]["warmup_steps"],
            max_steps = config["training"]["max_steps"],
            logging_steps = config["training"]["logging_steps"],
            output_dir = config["training"]["output_dir"],
            optim = config["training"]["optim"],
            seed = config["training"]["seed"],
        ),
    )
    
    trainer_stats = trainer.train()
    print("Training complete.")

    # 5. Inference (Verification)
    print("Running inference test...")
    FastLanguageModel.for_inference(model)
    messages = [
        {
            "role": "user",
            "content": "Mike is 30 years old, loves hiking and works as a coder."
        },
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    
    outputs = model.generate(input_ids=inputs, max_new_tokens=512, use_cache=True, temperature=0.7, do_sample=True, top_p=0.9)
    response = tokenizer.batch_decode(outputs)[0]
    print(f"Inference Response:\n{response}")

    # 6. Export
    if config["export"]["save_model"]:
        print(f"Exporting model to {config['export']['output_dir']}...")
        model.save_pretrained_gguf(
            config["export"]["output_dir"], 
            tokenizer, 
            quantization_method=config["export"]["quantization_method"],
            maximum_memory_usage=config["export"]["maximum_memory_usage"]
        )
        print("Export complete.")

if __name__ == "__main__":
    main()
