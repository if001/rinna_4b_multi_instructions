import json

import fire
import torch

from transformers import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

def load_model(base_model_name, lora_weight=None):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False
                                            # pad_token='<|pad|>'
                                            # pad_token='<|endoftext|>'
                                            )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        # config=config
    )

    if lora_weight:        
        model = PeftModel.from_pretrained(
            model,
            lora_weight,
            torch_dtype=torch.float16,
        )
        print('load lora...', lora_weight)
    model.eval()
    print('base model', base_model_name)
    
    return tokenizer, model

def generate_prompt_ja(convs):
    if len(convs) % 2 == 0:
      prompt = 'あなたはユーザーとして振る舞ってください。システムの応答に関する具体的な質問や関連する質問をします。\n\n'
    else:
      prompt = 'あなたはユーザーに親切でフレンドリーなAIです。ユーザーの質問に対し詳細に長い文章で質問に答えます。\n\n'
    
    for i, v in enumerate(convs):
      prefix = 'ユーザー'
      if i % 2 == 1:
        prefix = 'システム'
      prompt += f"{prefix}: {v}\n"

    if len(convs) % 2 == 0:
      prompt += 'ユーザー:'
    else:
      prompt += 'システム:'
    return prompt

def evaluate(
        tokenizer,
        model,
        prompt,        
        temperature=0.85,
        top_p=0.85,
        top_k=40,
        num_beams=1,
        min_tokens_len=16,
        **kwargs,
):
    device='cuda'
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    
    max_new_tokens = 256
    # input_len = input_ids.size()[-1]
    # min_tokens_len = min(input_len+min_tokens_len, max_new_tokens)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        #top_k=top_k,
        num_beams=num_beams,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=256,
            # min_new_tokens=min_tokens_len,
          )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return output


def gen_loop(
            tokenizer,
            model,
            first_text,
            max_cnt = 4,
            ):  
    convs = [first_text]
    cnt = 0
    
    while True:
        ## systemの返答を生成
        prompt = generate_prompt_ja(convs)
        output = evaluate(
                    tokenizer,
                    model,
                    prompt
        )
        system_output = output[len(prompt):]
        convs.append(system_output)

        ## userの返答を生成
        prompt = generate_prompt_ja(convs)
        output = evaluate(
                    tokenizer,
                    model,
                    prompt
        )
        user_output = output[len(prompt):]        
        convs.append(user_output)

        cnt += 1
        if max_cnt == cnt:
            break
    return convs


def main(
        base_model,        
        conv_file,
        output,
        lora_weight=None,
):
    gen_list = [
        "open_qa",
        "closed_qa",
        "brainstorming"
    ]
    with open(conv_file) as f:
        json_file = json.load(f)
    inner_loop_count = 10
    obj = {}
    tokenizer, model = load_model(base_model, lora_weight)
    for key in json_file.keys():
        if key not in gen_list:
            continue
        category_text = json_file[key]
        for i, text in enumerate(category_text):            
            for j in range(inner_loop_count):
                results = gen_loop(
                    tokenizer,
                    model,
                    text,
                    6
                )
                obj[f"{key}_{i}_{j}"] = results
                print(f'results {key} {i} {j}')
                for v in results:
                    print(v)
                print('-'*60)
    with open(output, 'w') as f:
        json.dump(obj, f, indent=2)

if __name__ == '__main__':
    fire.Fire(main)