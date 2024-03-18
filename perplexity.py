import evaluate
from prompts import *
from my_utils import *

model_name = "berkeley-nest/Starling-LM-7B-alpha"

perplexity = evaluate.load("perplexity", module_type="metric")

prompts = gsm8k_inference_prompts
original_test_dataset = read_jsonl('./data/gsm8k_test.jsonl')
testset = list(map(add_label, original_test_dataset))

input_dict = {}

for prompt in prompts:
    inputs = []
    for i, sample in enumerate(testset):
        question = sample['question']
        input_texts = f"GPT4 Correct User: Question: {question}\nAnswer: {prompt}<|end_of_turn|>GPT4 Correct Assistant:"
        inputs.append(input_texts)
    input_dict[prompt] = inputs


for prompt, input_texts in input_dict.items():

    results = perplexity.compute(model_id=model_name,
                                add_start_token=False,
                                predictions=input_texts)
    print(f"Prompt: {prompt} | Mean: {results['mean_perplexity']}")



