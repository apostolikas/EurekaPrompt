from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from my_utils import *
import random

original_test_dataset = read_jsonl('./data/gsm8k_test.jsonl')
testset = list(map(add_label, original_test_dataset))
random.seed(42)
samples = random.sample(testset, 50)

# model_name = 'berkeley-nest/Starling-LM-7B-alpha'
model_name = 'teknium/OpenHermes-2.5-Mistral-7B'
# model_name = "openchat/openchat_3.5"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16, device_map = 'auto')


fast_short_prompts = [
    "Find the correct answer as quickly as possible",
    "Answer the question correctly and as fast as you can",
    "Derive the solution swiftly and accurately",
    "Be quick and accurate in your calculations",
    "Your respond has to be both quick and precise",
]

slow_short_prompts = [
    "Slow down and think carefully before answering",
    "Take your time to think about the question",
    "Do not rush so you can provide the correct answer",
    "Think slowly so you can answer correctly",
    "Take your time to thoroughly think about the question",
]

fast_long_prompts = [
    "Make sure to answer the question correctly while also being quick, as time is of the essence",
    "Your response should be both quick and accurate, so make sure to think fast and answer correctly",
    "Be quick and accurate in your calculations, as speed and accuracy are equally important",
    "Focus on the question and swiftly find the correct answer, while also being accurate in your calculations",
    "Precision and speed are key in this problem, so make sure to answer the question correctly and quickly", 
]

slow_long_prompts = [
    "Take your time to think about the question and provide the correct answer, as accuracy is more important than speed",
    "Do not rush so you can provide the correct answer, as accuracy is more important than speed",
    "Think slowly so you can answer correctly, as accuracy is more important than speed",
    "Take your time to thoroughly think about the question and provide the correct answer, as accuracy is more important than speed",
    "Make sure to answer the question correctly, so take your time and think about the question thoroughly",
]
def prompt_time(prompts):

    for prompt in prompts:
        print("Measuring time for prompt: ", prompt)
        times = {}

        for i, sample in enumerate(samples):

            question = sample['question']
            label = sample['label']
            model_input = f'''Q: {question}\nA: {prompt}'''

            if model_name == 'berkeley-nest/Starling-LM-7B-alpha' or model_name == 'openchat/openchat_3.5':
                input_prompt = f'''GPT4 Correct User: {model_input}<|end_of_turn|>GPT4 Correct Assistant:'''
                encoding_start = time.time()
                input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
                encoding_end = time.time()
                generating_start = time.time()
                outputs = model.generate(input_ids, max_new_tokens=250, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                generating_end = time.time()
                response_ids = outputs[0]
                decoding_start = time.time()
                text_output = tokenizer.decode(response_ids, skip_special_tokens=True)
                decoding_end = time.time()
                text_output = text_output.split("GPT4 Correct Assistant:")[1]

            elif model_name == 'teknium/OpenHermes-2.5-Mistral-7B':
                messages = [{"role": "user", "content": model_input},]
                encoding_start = time.time()
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to('cuda')
                encoding_end = time.time()
                generating_start = time.time()
                generated_ids = model.generate(gen_input, max_new_tokens = 250, pad_token_id=tokenizer.eos_token_id)
                generating_end = time.time()
                decoding_start = time.time()
                decoded = tokenizer.batch_decode(generated_ids)
                decoding_end = time.time()
                text_output = decoded[0]
                lines = text_output.split('\n')  
                found_A = False
                modified_string = ''
                for line in lines:
                    if found_A:
                        modified_string += line + '\n'  
                    elif line.startswith('A:'):
                        found_A = True
                    text_output = modified_string.replace("<|im_end|>", "")

            else:
                raise ValueError("Model not supported")

            encoding_time = encoding_end - encoding_start
            generating_time = generating_end - generating_start
            decoding_time = decoding_end - decoding_start

            eval_result = evaluate_GSM8K(text_output, label)

            times[i] = {
                'encoding_time': encoding_time,
                'generating_time': generating_time,
                'decoding_time': decoding_time,
                'total_time': encoding_time + generating_time + decoding_time,
                'eval_result': eval_result
            }

            # print(f"Sample {i+1} : Correct = {eval_result}")
            # print(f"Encoding time: {encoding_time}")
            # print(f"Generating time: {generating_time}")
            # print(f"Decoding time: {decoding_time}")
            # print(f"Total time: {encoding_time + generating_time + decoding_time}")
            # print("")

        avg_encoding_time = sum([times[i]['encoding_time'] for i in times])/len(times)
        avg_generating_time = sum([times[i]['generating_time'] for i in times])/len(times)
        avg_decoding_time = sum([times[i]['decoding_time'] for i in times])/len(times)
        avg_total_time = sum([times[i]['total_time'] for i in times])/len(times)

        print(f"Avg times for prompt: {prompt}")
        print("Encoding time: ", sum([times[i]['encoding_time'] for i in times])/len(times))
        print("Generating time: ", sum([times[i]['generating_time'] for i in times])/len(times))
        print("Decoding time: ", sum([times[i]['decoding_time'] for i in times])/len(times))
        print("Total time: ", sum([times[i]['total_time'] for i in times])/len(times))
        print("Correct answers: ", sum([times[i]['eval_result'] for i in times]), "/", len(times))
        print("")

    return times, avg_encoding_time, avg_generating_time, avg_decoding_time, avg_total_time

_, fast_short_encoding_time, fast_short_generating_time, fast_short_decoding_time, fast_short_total_time = prompt_time(fast_short_prompts)
_, slow_short_encoding_time, slow_short_generating_time, slow_short_decoding_time, slow_short_total_time = prompt_time(slow_short_prompts)
_, fast_long_encoding_time, fast_long_generating_time, fast_long_decoding_time, fast_long_total_time = prompt_time(fast_long_prompts)
_, slow_long_encoding_time, slow_long_generating_time, slow_long_decoding_time, slow_long_total_time = prompt_time(slow_long_prompts)

print(f"Fast & short prompts")
print(f"Encoding time: {fast_short_encoding_time}")
print(f"Generating time: {fast_short_generating_time}")
print(f"Decoding time: {fast_short_decoding_time}")

print(f"Slow & short prompts")
print(f"Encoding time: {slow_short_encoding_time}")
print(f"Generating time: {slow_short_generating_time}")
print(f"Decoding time: {slow_short_decoding_time}")

print(f"Fast & long prompts")
print(f"Encoding time: {fast_long_encoding_time}")
print(f"Generating time: {fast_long_generating_time}")
print(f"Decoding time: {fast_long_decoding_time}")

print(f"Slow & long prompts")
print(f"Encoding time: {slow_long_encoding_time}")
print(f"Generating time: {slow_long_generating_time}")
print(f"Decoding time: {slow_long_decoding_time}")

