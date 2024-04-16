from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from my_utils import *
from prompts import *
from tqdm import tqdm

def create_model_input(task, question, instruction, choices, narrative):

    if task in ['gsm8k', 'svamp']:
        model_input = f"GPT4 Correct User: Q:{question}\nA: {instruction}<|end_of_turn|>GPT4 Correct Assistant:"

    elif task in ['csqa', 'social_iqa', 'sports_und', 'date_under', 'causal_judg']:
        input_prompt = f'''Question: {question}\nAnswer Choices: {choices}\nAnswer: {instruction}'''
        model_input = f'''GPT4 Correct User: {input_prompt}<|end_of_turn|>GPT4 Correct Assistant:'''

    elif task == 'abs_nar':
        input_prompt = f'''Question: Can you choose the most related proverb from the list of 5 proverbs given a narrative?\nNarrative: {narrative}\nAnswer choices: {choices}\nAnswer: {instruction}'''
        model_input = f'''GPT4 Correct User: {input_prompt}<|end_of_turn|>GPT4 Correct Assistant:'''

    return model_input

def generate_response(task, decode_strategy, model, tokenizer, question, instruction, choices=None, narrative=None):

    model_input = create_model_input(task, question, instruction, choices, narrative)
    
    inputs = tokenizer(model_input, return_tensors="pt").to('cuda')

    if decode_strategy == 'greedy':
        outputs = model.generate(**inputs, do_sample = False, num_beams = 1, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'contrastive_search':
        outputs = model.generate(**inputs, penalty_alpha=0.6, top_k=4, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'multinomial_sampling':
        outputs = model.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'beam_search':
        outputs = model.generate(**inputs, num_beams=5, do_sample = False, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'beam_search_with_multinomial_sampling':
        outputs = model.generate(**inputs, num_beams=5, do_sample=True, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'top_k_sampling':
        outputs = model.generate(**inputs, do_sample=True, top_k=50, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    elif decode_strategy == 'top_p_sampling':
        outputs = model.generate(**inputs, do_sample=True, top_p=0.9, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'sampling0.25':
        outputs = model.generate(**inputs, do_sample=True, top_k = 0, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, temperature=0.25)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    elif decode_strategy == 'sampling0.5':
        outputs = model.generate(**inputs, do_sample=True, top_k = 0, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, temperature=0.5)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    elif decode_strategy == 'sampling0.75':
        outputs = model.generate(**inputs, do_sample=True, top_k = 0, max_new_tokens = 600, pad_token_id = tokenizer.pad_token_id, eos_token_id = tokenizer.eos_token_id, temperature=0.75)
        generated_text= tokenizer.batch_decode(outputs, skip_special_tokens=True)

    else:
        raise ValueError("Invalid decoding strategy")
    
    return generated_text

if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", device_map='auto', torch_dtype=torch.float16)
    decode_strategies = ["greedy", "contrastive_search", "multinomial_sampling", "beam_search", "beam_search_with_multinomial_sampling", "top_k_sampling", "top_p_sampling", "sampling0.25", "sampling0.5", "sampling0.75"]

    tasks = ['sports_und']#, 'csqa', 'abs_nar', 'causal_judg', 'social_iqa', 'date_under', 'sports_und']
    for task in tasks:
        num_of_samples = 50 #if task in ['gsm8k','svamp','csqa'] else 35

        _, testset = load_data(task)
        prompts = load_inference_prompts(task)
        random.seed(0)
        samples = random.sample(testset, num_of_samples)
        file_name = f"./decoded/decode_results_{task}.txt"

        with open(file_name, 'w') as f:

            for prompt in prompts:
                print(f"Decoding for prompt: {prompt}")
                for decode_strategy in decode_strategies:
                    for i, sample in tqdm(enumerate(samples)):
                        responses = []
                        if task == 'gsm8k':
                            question = sample['question']
                            response_text = generate_response(task, decode_strategy, model, tokenizer, question, prompt)

                        elif task == 'svamp':
                            question = sample['full_question']
                            response_text = generate_response(task, decode_strategy, model, tokenizer, question, prompt)

                        elif task == 'csqa':
                            question = sample['question']['stem']
                            choices = sample['choice_answers']
                            response_text = generate_response(task, decode_strategy, model, tokenizer, question, prompt, choices)

                        elif task == 'abs_nar':
                            narrative = sample['input']
                            choices = sample['answer_choices']
                            response_text = generate_response(task, decode_strategy, model, tokenizer, question = None, instruction = prompt, choices = choices, narrative = narrative)

                        elif task in ['causal_judg', 'social_iqa', 'date_under', 'sports_und']:
                            question = sample['input']
                            answer_choices = sample['answer_choices']
                            response_text = generate_response(task, decode_strategy, model, tokenizer, question = question, instruction = prompt , choices = answer_choices, narrative = None)

                        f.write(f"Prompt: {prompt} | Decode Strategy: {decode_strategy} | Sample: {i} | Output: {response_text}\n")







