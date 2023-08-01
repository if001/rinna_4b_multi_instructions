from typing import List
import sys

import fire
import torch
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from utils.prompter import Prompter

VAL_SET_SIZE = 2000



def train(
        base_model: str = "",
        data_path: str = "",
        output_dir: str = "",
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 256,
        val_set_size: int = 2000,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = ["query_key_value"],
        prompt_template_name: str = "alpaca_ja",  # The prompt template to use, will default to alpaca.,
        verbose: bool = False
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"verbose: {verbose}\n"
    )
    device_map = 'auto'

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,        
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config,
        #offload_folder="offload",
        #offload_state_dict = True,
    )
    model = prepare_model_for_int8_training(model)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    print('special token: ', tokenizer.special_tokens_map)
    print('pad:', tokenizer.pad_token)
    print('eos:', tokenizer.eos_token)
    tokenizer.padding_side = "right"  # Allow batched inference

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()



    ## --- data set ---
    prompter = Prompter(prompt_template_name, verbose=verbose)
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt_conv(data_point):
        data = []
        prefix = "以下はユーザーとアシスタントの会話です。アシスタントは親切で丁寧に詳細を回答します。\n\n"
        print('gen')
        for conversations in data_point["conversations"]:            
            for i in range(len(conversations)):                
                prompt = prefix
                for j, v in enumerate(conversations[:i+1]):
                    prompt += "### ユーザー: \n" + v["S"] + '\n\n' + "### アシスタント: \n" + v["U"] + '</s>'
                    if j != i:
                        prompt += '\n\n'
                    #print(prompt)
                    #print('-'*20)
                    while '\n' in prompt:
                        prompt = prompt.replace('\n', '<NL>')

                    tokenized_prompt = tokenize(prompt, add_eos_token=True)
                    #print('tokenizered ', tokenized_prompt)
                    #print('='*20)
                    data.append(tokenized_prompt)
        return data
    
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )        
        tokenized_full_prompt = tokenize(full_prompt)
        if verbose:
            print('full: ', full_prompt)
            print('tokenized: ', tokenized_full_prompt)
            print('-'*60)
        # if not train_on_inputs:
        #     user_prompt = prompter.generate_prompt(
        #         data_point["instruction"], data_point["input"]
        #     )
        #     tokenized_user_prompt = tokenize(
        #         user_prompt, add_eos_token=add_eos_token
        #     )
        #     user_prompt_len = len(tokenized_user_prompt["input_ids"])

        #     if add_eos_token:
        #         user_prompt_len -= 1

        #     tokenized_full_prompt["labels"] = [
        #         -100
        #     ] * user_prompt_len + tokenized_full_prompt["labels"][
        #         user_prompt_len:
        #     ]  # could be sped up, probably
        return tokenized_full_prompt

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
    )

    train_val = data["train"].train_test_split(
        test_size=val_set_size, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )

    ## train for conv data
    # train_data = generate_and_tokenize_prompt_conv(train_val["train"].shuffle())
    # print("train_data len", len(train_data))
    # val_data = generate_and_tokenize_prompt_conv(train_val["test"].shuffle())
    # print("val_data", len(val_data))
    # print("train_data", train_data[0])

    train_data = Dataset.from_list(train_data)
    val_data = Dataset.from_list(val_data)    
    print("train_data len", len(train_data))
    print("val_data", len(val_data))
    ## --- data set ---

    gradient_accumulation_steps = batch_size // micro_batch_size
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    #old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    trainer.save_model(output_dir)
    # model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)